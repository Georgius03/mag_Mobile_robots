import math, csv, socket, requests
import heapq

import cv2
import cv2.aruco as aruco
import numpy as np

from src.core.config import settings

class Logger:
    def __init__(self, filename: str) -> None:
        self.filename = filename  # Сохраняем имя файла
        self.log_file = None
        self.csv_writer = None
        self.reset_and_start() # Инициализируем при создании
        
        print("Logger initialised!")

    def reset_and_start(self) -> None:
        """Закрывает старый файл и начинает запись в новый (перезапись)"""
        if self.log_file is not None:
            self.log_file.close()
        
        # Режим "w" очищает файл при открытии
        self.log_file = open(self.filename, "w", newline="")
        self.init_writer()

    def init_writer(self) -> None:
        if self.log_file is not None:
            self.csv_writer = csv.writer(self.log_file)
            self.csv_writer.writerow(["time_s",
                                    "pos_x_px", "pos_y_px",
                                    "vx_px_s", "vy_px_s",
                                    "speed_px_s",
                                    "target_vx_px_s", "target_vy_px_s",
                                    "target_speed_px_s",])
            # Сбрасываем буфер на диск, чтобы заголовок записался сразу
            self.log_file.flush()

    def write_log(self, t: float, pos: np.ndarray, vel: np.ndarray, target_vel: np.ndarray) -> None:
        speed: float = float(np.linalg.norm(vel))
        target_speed: float = float(np.linalg.norm(target_vel))
        if self.csv_writer is not None:
            self.csv_writer.writerow([
                f"{t:.6f}",
                f"{pos[0]:.3f}", f"{pos[1]:.3f}",
                f"{vel[0]:.6f}", f"{vel[1]:.6f}",
                f"{speed:.6f}",
                f"{target_vel[0]:.6f}", f"{target_vel[1]:.6f}",
                f"{target_speed:.6f}"
            ])
    
    def close(self) -> None:
        if self.log_file:
            self.log_file.close()

class CameraProcessor:
    def __init__(self) -> None:
        self.config = settings
        
        # Переменные преобразований
        self.ORIG_SIZE: tuple[int, int] = (self.config.camera.cam_shape_width, self.config.camera.cam_shape_height)
        
        self.OUTPUT_SIZE: tuple[int, int] = (self.config.map_params.resolution, self.config.map_params.resolution)
        self.DST_POINTS: np.ndarray = np.array(
            [
                [0, 0],
                [self.config.map_params.resolution, 0],
                [self.config.map_params.resolution, self.config.map_params.resolution],
                [0, self.config.map_params.resolution],
            ], dtype=np.float32
        )
        
        self.SRC_POINTS: np.ndarray = np.array(
            [
                self.config.map_params.left_up,
                self.config.map_params.right_up,
                self.config.map_params.right_down,
                self.config.map_params.left_down,
            ], dtype=np.float32
        )
        
        self.perspective_matrix: np.ndarray = np.array([[1, 0, 100],
                                                        [1, 0, 100]], dtype=np.float32)
        self.kernel_denoise: np.ndarray = np.ones((10, 10), np.uint8)
        self.robot_radius_dilatation_pixels: int = self.config.grid.robot_radius

        self.aruco_6x6 = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(aruco.DICT_6X6_50),
            aruco.DetectorParameters()
        )

        # Изображение для обработки и отображения
        self.frame: np.ndarray = np.zeros((self.ORIG_SIZE[1], self.ORIG_SIZE[0], 3), dtype=np.uint8)
        self.display: np.ndarray = np.zeros((self.OUTPUT_SIZE[0], self.OUTPUT_SIZE[1], 3), dtype=np.uint8)
        self.mask: np.ndarray = np.zeros((self.OUTPUT_SIZE[0], self.OUTPUT_SIZE[1]), dtype=np.uint8)
        self.pooled_mask: np.ndarray = np.zeros((self.OUTPUT_SIZE[0], self.OUTPUT_SIZE[1]), dtype=np.uint8)
        
        # Переменные окружения
        self.system_calibrated_flag: bool = False
        self.trans_center: tuple[int, int] = (1000, 1000)

        print("CameraProcessor initialised!")

    # Загрузка изображения
    def load_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            raise FileNotFoundError(f"Image not found = None")
        
        frame = self._frame_preprocessing(frame)
        
        if self.system_calibrated_flag:
            frame = cv2.warpPerspective(
                frame.copy(),
                self.perspective_matrix,
                self.OUTPUT_SIZE
            )
            self.frame = frame.copy()
        else:
            self.frame = frame.copy()
        self.display = frame.copy()
        
        if self.system_calibrated_flag:
            self._find_obstacles()
            self._draw_obstacles()
    
    # Предобработка кадров (выравнивание гистограммы, резкость)
    def _frame_preprocessing(self, frame: np.ndarray) -> np.ndarray:
        if self.config.camera.equalize_hist:
            frame = self._equalize_histogram(frame)
        
        if self.config.camera.sharpening:
            frame = self._sharpening(frame)
        
        return frame
    
    # Выравнивание гистограммы
    def _equalize_histogram(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        # Выравнивание гистограммы только для канала Y (яркость)
        frame[:,:,0] = cv2.equalizeHist(frame[:,:,0])
        # Обратное преобразование в BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
        return frame
    
    # Резкость изображения
    def _sharpening(self, frame: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        return sharpened
    
    # Калибровка системы по точкам
    def calibrate_system(
        self,
        do_calibrate: bool = False
    ) -> None:
        for y, x in self.SRC_POINTS:
            cv2.circle(
                self.display,
                (int(x), int(y)),
                5,
                (0, 0, 255),
                -1
            )
        try:
            if do_calibrate:
                self.perspective_matrix = self._compute_perspective()
                self.trans_center = self._compute_projected_center()
                self.system_calibrated_flag = True
                print("Калибровка выполнена")

        except ValueError as e:
            print(e)
    
    # Расчёт матрицы преобразования перспективы
    def _compute_perspective(
        self,
    ) -> np.ndarray:
        """
        Выполняет калибровку полигона по 4 точкам.
        """
        
        return cv2.getPerspectiveTransform(self.SRC_POINTS[:, ::-1], self.DST_POINTS)
    
    # Расчёт положения спроецированного исходного центра изображения (для коррекции маски)
    def _compute_projected_center(
        self,
    ) -> tuple[int, int]:
        """
        Возвращает положение оптического центра камеры
        после применения warpPerspective.
        """

        original_center: np.ndarray = np.array(
            [[[self.ORIG_SIZE[0] / 2.0, self.ORIG_SIZE[1] / 2.0]]],
            dtype=np.float32
        )

        warped_center: np.ndarray = cv2.perspectiveTransform(
            original_center,
            self.perspective_matrix
        )

        return warped_center[0][0][::-1]
    
    # Выделение препятствий на изображении
    def _find_obstacles(self) -> None:
        # --- HSV ---
        hsv: np.ndarray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        if self.config.camera.equalize_hist:
            mask: np.ndarray = cv2.inRange(hsv, np.array([0, 170, 60]), np.array([179, 255, 255]))
        else:
            mask: np.ndarray = cv2.inRange(hsv, np.array([0, 100, 135]), np.array([179, 255, 255]))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_denoise)
        mask = cv2.erode(mask, kernel=self.kernel_denoise, iterations=2)
        mask = cv2.dilate(mask, kernel=self.kernel_denoise, iterations=2)

        inv_mask = cv2.bitwise_not(mask)
        dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
        expanded = (dist <= self.robot_radius_dilatation_pixels).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, expanded)
        
        mask = self._project_floor_mask(
            mask,
            H=self.config.camera.H,
            h=self.config.camera.h,
            center=self.trans_center,
            )
        
        WALL_WIDTH = self.config.grid.WALL_WIDTH
        mask[:WALL_WIDTH, :] = 255
        mask[-WALL_WIDTH:, :] = 255
        mask[:, :WALL_WIDTH] = 255
        mask[:, -WALL_WIDTH:] = 255
        
        self.mask = mask
        
        self.pooled_mask = self._maxpool2D(
            inflated_mask=mask,
            grid_step=self.config.grid.step
        )
    
    # Проекция маски на поверхность с учётом камеры
    def _project_floor_mask(
        self,
        mask: np.ndarray,
        H: float,
        h: float,
        center: tuple[float, float],
    ) -> np.ndarray:
        """
        Проекция маски на плоскость полигона.
        """
        k: float = (H - h) / H
        cx, cy = center

        M: np.ndarray = np.array([
            [k, 0, cx * (1 - k)],
            [0, k, cy * (1 - k)]
        ], dtype=np.float32)

        return cv2.warpAffine(mask, M, mask.shape[::-1], flags=cv2.INTER_NEAREST)
    
    # Пулинг маски для уменьшения дискретизации
    def _maxpool2D(
        self,
        inflated_mask: np.ndarray,
        grid_step: int
    ) -> np.ndarray:
        """
        Формирует дискретную карту занятости через блочное
        max-пулинг уменьшение.

        Args:
            inflated_mask: 255 = препятствие
            grid_step: шаг дискретизации [px]

        Returns:
            occupancy_grid:
                0 = занято
                1 = свободно
        """

        height, width = inflated_mask.shape

        new_h: int = height // grid_step
        new_w: int = width // grid_step

        trimmed: np.ndarray = inflated_mask[
            :new_h * grid_step,
            :new_w * grid_step
        ]

        reshaped: np.ndarray = trimmed.reshape(
            new_h,
            grid_step,
            new_w,
            grid_step
        )

        # max по блоку → если есть хоть один 255
        pooled: np.ndarray = reshaped.max(axis=(1, 3))

        occupancy_grid: np.ndarray = (pooled == 0).astype(np.uint8)

        return occupancy_grid
    
    # Отрисовка обнаруженных препятствий на изображении
    def _draw_obstacles(self) -> None:
        
        visualize_pool_mask = cv2.resize(
            self.pooled_mask,
            self.OUTPUT_SIZE,
            interpolation=cv2.INTER_NEAREST
        )
        
        pooled_contours, _ = cv2.findContours(
            visualize_pool_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours, _ = cv2.findContours(
            self.mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(self.display, pooled_contours, -1, (255, 0, 0), 2)
        cv2.drawContours(self.display, contours, -1, (0, 255, 0), 2)
    
    # Трекинг робота
    def get_robot_attitude(self) -> tuple[np.ndarray, float] | None:
        corners, ids, _ = self.aruco_6x6.detectMarkers(self.frame)

        center_x, center_y, angle = 0, 0, 0
        
        if ids is not None:
            c: np.ndarray = corners[0][0]

            center_x: int = int(np.mean(c[:, 1]))
            center_y: int = int(np.mean(c[:, 0]))

            dx: float = c[1][0] - c[0][0]
            dy: float = c[1][1] - c[0][1]

            angle: float = math.atan2(dx, dy)
        
            pose_2D: np.ndarray = np.array([center_x, center_y], dtype=int)
            pose_2D = self._project_floor_point(
                point=pose_2D,
                H=self.config.camera.H,
                h=self.config.camera.h,
                center=self.trans_center
            )
        
            return (pose_2D.astype(np.int16), angle)
        else:
            return None
    
    # Проецирование точки на поверхность с учётом камеры
    def _project_floor_point(
        self,
        point: np.ndarray,
        H: float,
        h: float,
        center: tuple[float, float]
    ) -> np.ndarray:
        """
        Проецирует точку на плоскость полигона
        аналогично функции project_mask().

        Args:
            point: np.ndarray shape (2,) – (x, y) [px]
            H: высота камеры [px]
            h: высота объекта [px]
            center: центр проекции (cx, cy)

        Returns:
            np.ndarray shape (2,) – спроецированная точка [px]
        """
        k: float = (H - h) / H
        cx, cy = center

        x_proj: float = k * point[0] + cx * (1 - k)
        y_proj: float = k * point[1] + cy * (1 - k)

        return np.array([x_proj, y_proj], dtype=np.int32)

    # Вернуть обработанное изображение
    def get_display(self) -> np.ndarray:
        return self.display.copy()
    
    # Вернуть обработанное изображение
    def get_mask(self) -> np.ndarray:
        return self.mask.copy()

class RobotinoUnit:
    def __init__(self) -> None:
        self.config = settings
        self.IP_ADDRESS: str = self.config.socket_params.IP_ADDRESS
        self.PORT: int = self.config.socket_params.PORT
        
        self.server_enable = self.config.socket_params.enable
        if self.server_enable:
            self.robotino_socket: (socket.socket | None) = self._connect_to_robotino()
        
        self.curr_pose2D: np.ndarray = np.array([0, 0])
        self.curr_angle: float = 0.0
        
        self.velocity: np.ndarray = np.array([0.0, 0.0])
        
        # self.comp_matrix = np.array([[0, -1],
        #                              [1, 0]])
        
        self.comp_matrix = np.array([[0, -1],
                                     [1, 0]])
        
        print("RobotinoUnit initialised!")
    
    # Управление по направлению к целевой точке
    def navigate(self, target_pose2D) -> None:
        velocity, is_reached = self._compute_velocity(
                    current_position=self.curr_pose2D,
                    target_position=target_pose2D,
                    v_max=self.config.move.max_speed,
                    tolerance=self.config.move.dist_stop,
                    k_p=self.config.move.k_prop
                )
        
        velocity = self._rotate_vector(velocity)
        self.velocity = velocity
        
        self._send_velocity(velocity[0], velocity[1], 0)
    
    # Управление по скоростям
    def navigate_velocity(self, velocity, omega=0) -> None:
        velocity = self._rotate_vector(velocity)
        self.velocity = velocity
        
        self._send_velocity(velocity[0], velocity[1], omega)
    
    # Обновление текущего положения робота
    def update_state(self, curr_pose: np.ndarray, angle: float) -> None:
        self.curr_pose2D = curr_pose
        self.curr_angle = angle

    # Поворот вектора скорости с учётом ориентации полигона и поворота робота
    def _rotate_vector(self, vector: np.ndarray) -> np.ndarray:
        rotation_matrix = np.array([[math.cos(self.curr_angle), -math.sin(self.curr_angle)],
                                     [math.sin(self.curr_angle), math.cos(self.curr_angle)]])
        return vector @ self.comp_matrix @ rotation_matrix
    
    # Подключение к Robotino
    def _connect_to_robotino(self) -> (socket.socket | None):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.IP_ADDRESS, self.PORT))
            print("Successfully connected")
            return sock
        
        except Exception as e:
            print(f"Error connecting: {e}")
            return None
    
    # Отключение от Robotino
    def disconnect(self) -> None:
        if self.server_enable:
            if self.robotino_socket is not None:
                self.robotino_socket.close()
    
    # Отправка управляющих сигналов
    def _send_velocity(self, vx: float, vy: float, omega: float):
        if self.server_enable:
            url = f"http://{self.IP_ADDRESS}/data/omnidrive"
            data = [float(vx), float(vy), float(omega)]
            try:
                response = requests.post(url, json=data, timeout=0.05)
                if response.status_code == 200:
                    # print(f"Sent Vx: {vx}, Vy: {vy}, w: {omega}")
                    pass
                else:
                    print(f"Send error: {response.status_code} - {response.text}")
                    pass
            except Exception as e:
                print(f"Error sending data: {e}")

    # Расчёт результирующего вектора скорости робота
    def _compute_velocity(
        self,
        current_position: np.ndarray,
        target_position: np.ndarray,
        v_max: float,
        tolerance: float,
        k_p: float = 1
    ) -> tuple[np.ndarray, bool]:
        """
        Вычисляет вектор линейной скорости движения к целевой точке
        в декартовой системе координат.

        Args:
            current_position : np.ndarray shape (2,) – текущая позиция [px]
            target_position  : np.ndarray shape (2,) – целевая позиция [px]
            v_max            : float – максимальная скорость [px/s]
            tolerance        : float – радиус достижения цели [px]
            k_p              : float – коэффициент пропорционального регулятора [1/s]

        Returns:
            velocity : np.ndarray shape (2,) – вектор скорости [px/s]
            reached  : bool – флаг достижения цели
        """

        # --- Вектор ошибки ---
        error: np.ndarray = target_position.astype(np.float32) - \
                            current_position.astype(np.float32)

        distance: float = float(np.linalg.norm(error))

        # --- Проверка достижения цели ---
        if distance <= tolerance:
            return np.zeros(2, dtype=np.float32), True

        # --- Пропорциональный регулятор ---
        velocity: np.ndarray = k_p * error / 1000

        speed: float = float(np.linalg.norm(velocity))

        # --- Ограничение максимальной скорости ---
        if speed > v_max and speed > 1e-6:
            velocity = velocity / speed * v_max

        return velocity.astype(np.float32), False

class AStarPlanner:
    def __init__(self) -> None:
        self.config = settings
        self.path: list[tuple[int, int]] = []
        self.grid_step: int = self.config.grid.step

        print("AStarPlanner initialised!")
        
    # Расчёт расстояния между точками старта и целью
    def _heuristic(
        self,
        a: tuple[int, int],
        b: tuple[int, int]
    ) -> float:
        """
        Вычисляет эвристическую оценку расстояния между узлом a и целью b.

        Используется евклидова метрика:

            h(n) = sqrt((y1 - y2)^2 + (x1 - x2)^2)

        ВАЖНО:
        Эвристика не учитывает препятствия.
        Это нижняя граница реального пути, что гарантирует оптимальность A*.
        """
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    # Восстановление пути
    def _reconstruct_path(
        self,
        came_from: dict[tuple[int, int], tuple[int, int]],
        start: tuple[int, int],
        goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        Восстанавливает путь от goal к start
        с использованием словаря предков came_from.

        Путь строится в обратном порядке:
            goal → ... → start
        затем разворачивается.

        Возвращает:
            Список координат [(y, x), ...] от старта к цели.
            Если путь не найден — пустой список.
        """

        # Если цель не имеет предка — путь не построен
        if goal not in came_from:
            return []

        path: list[tuple[int, int]] = [goal]
        current: tuple[int, int] = goal

        # Двигаемся назад по дереву поиска
        while current != start:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path
    
    # Алгоритм A*
    def astar(
        self,
        pooled_mask: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        Реализация алгоритма A* для двумерной карты.

        pooled_mask:
            np.ndarray формы (H, W)
            0 — препятствие
            1 — свободная область

        start, goal:
            Координаты (y, x)

        Возвращает:
            Список координат пути.
            Если путь невозможен — пустой список.
        """

        height, width = pooled_mask.shape

        # --- Проверка допустимости старта и цели ---
        if pooled_mask[start] == 0:
            return []
        
        if not (0 <= start[0] < height and 0 <= start[1] < width):
            return []
        if not (0 <= goal[0] < height and 0 <= goal[1] < width):
            return []

        if pooled_mask[goal] == 0:
            return []

        # --- Очередь с приоритетом ---
        # Храним (f_cost, координата)
        open_set: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start))

        # Словарь предков для восстановления пути
        came_from: dict[tuple[int, int], tuple[int, int]] = {}

        # g_cost хранит реальную стоимость пути от старта
        g_cost: dict[tuple[int, int], float] = {
            start: 0.0
        }

        # 8-связная окрестность
        directions = [
            (-1, 0), (1, 0),      # вверх, вниз
            (0, -1), (0, 1),      # влево, вправо
            (-1, -1), (-1, 1),    # диагонали
            (1, -1), (1, 1)
        ]

        while open_set:

            # Извлекаем узел с минимальным f = g + h
            _, current = heapq.heappop(open_set)

            # Если достигли цели — восстанавливаем путь
            if current == goal:
                self.path = self._reconstruct_path(came_from, start, goal)
                return self.path

            # Проверяем всех соседей
            for dy, dx in directions:

                ny: int = current[0] + dy
                nx: int = current[1] + dx

                # Проверка выхода за границы карты
                if not (0 <= ny < height and 0 <= nx < width):
                    continue

                # Проверка препятствия
                if pooled_mask[ny, nx] == 0:
                    continue

                # Стоимость перехода:
                # 1 для прямых ходов
                # sqrt(2) для диагоналей
                move_cost: float = float(np.hypot(dy, dx))

                tentative_g: float = g_cost[current] + move_cost
                neighbor: tuple[int, int] = (ny, nx)

                # Если путь к соседу найден впервые
                # или найден более короткий путь
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:

                    g_cost[neighbor] = tentative_g
                    f_cost: float = tentative_g + self._heuristic(neighbor, goal)

                    heapq.heappush(open_set, (f_cost, neighbor))
                    came_from[neighbor] = current

        # Если очередь пуста — путь не существует
        return []

    # Визуализация пути
    def draw_path(
        self,
        image: np.ndarray,
        point_radius: int = 20,
        line_thickness: int = 5,
        circle_color: tuple = (128, 0, 255),
        line_color: tuple = (0, 128, 255),
    ) -> None:
        """
        Отрисовывает путь робота поверх изображения:
        - узлы (круги)
        - соединяющие сегменты (линии)

        :param image: рабочее изображение (BGR)
        :param step: размер клетки сетки (px)
        :param point_radius: радиус отображаемых точек (px)
        :param line_thickness: толщина линии (px)
        :param circle_color: цвет точек (BGR)
        :param line_color: цвет линий (BGR)
        """

        if len(self.path) < 2:
            return

        pixel_points: list[tuple[int, int]] = []

        # --- преобразование grid → pixel ---
        for node in self.path:
            pixel_point: tuple[int, int] = self._grid_to_pixel(node)
            # OpenCV: (x, y)
            pixel_points.append(pixel_point[::-1])

        # --- отрисовка линий ---
        for i in range(len(pixel_points) - 1):
            cv2.line(
                image,
                pixel_points[i],
                pixel_points[i + 1],
                line_color,
                line_thickness
            )

        # --- отрисовка узлов ---
        for point in pixel_points:
            cv2.circle(
                image,
                point,
                point_radius,
                circle_color,
                -1
            )

    # Вывод информации о пути
    def print_path_info(
        self,
    ) -> None:
        """
        Выводит инфографику построенного пути.
        """
        
        if len(self.path) == 0:
            print("\n===== PATH INFO =====")
            print("Path: NOT FOUND")
            print("=====================\n")
            return
        
        # --- Старт и цель ---
        start_node: tuple[int, int] = self.path[0]
        goal_node: tuple[int, int] = self.path[-1]
        
        start_px: tuple[int, int] = self._grid_to_pixel(start_node)
        goal_px: tuple[int, int] = self._grid_to_pixel(goal_node)
        
        # --- Длина пути по сетке ---
        length_grid: float = 0.0
        for i in range(1, len(self.path)):
            dy: int = self.path[i][0] - self.path[i - 1][0]
            dx: int = self.path[i][1] - self.path[i - 1][1]
            length_grid += float(np.hypot(dy, dx))
        
        # --- Перевод в пиксели ---
        length_px: float = length_grid * self.grid_step
        
        # --- Прямое расстояние ---
        direct_dist_px: float = float(
            np.linalg.norm(np.array(goal_px) - np.array(start_px))
        )
        
        curvature_ratio: float = (
            length_px / direct_dist_px
            if direct_dist_px > 1e-6 else 0.0
        )
        
        # --- Инфографика ---
        print("\n========== PATH INFO ==========")
        print(f"Nodes count        : {len(self.path)}")
        print(f"Start (px)         : ({start_px[0]:.1f}, {start_px[1]:.1f})")
        print(f"Goal  (px)         : ({goal_px[0]:.1f}, {goal_px[1]:.1f})")
        print("--------------------------------")
        print(f"Path length (grid) : {length_grid:.3f} cells")
        print(f"Path length (px)   : {length_px:.3f} px")
        print(f"Direct distance    : {direct_dist_px:.3f} px")
        print("--------------------------------")
        print(f"Tortuosity         : {curvature_ratio:.3f}")
        print("================================\n")
    
    # Пересчёт положения точки пути на центр дискретизированного пикселя
    def _grid_to_pixel(
        self,
        node: tuple[int, int],
    ) -> tuple[int, int]:
        
        return (node[0] * self.grid_step + self.grid_step // 2,
                node[1] * self.grid_step + self.grid_step // 2)

# class SplineController:
#     def __init__(self, config: dict) -> None:
#         self.config = settings

#         self.path: list[tuple[int, int]] = []

#         # Параметры сплайна
#         self.points: np.ndarray | None = None
#         self.arc_lengths: np.ndarray | None = None
#         self.total_length: float = 0.0

#         # Текущее положение по длине дуги
#         self.current_s: float = 0.0
#         self.grid_step: int = config.grid.step

#         print("SplineController initialised!")

#     # ================= Загрузка пути =================
#     def load_path(self, path: list[tuple[int, int]]) -> None:
#         self.path = path.copy()
#         self._prepare_spline()

#     # ================= Подготовка сплайна =================
#     def _prepare_spline(self) -> None:
#         if len(self.path) < 2:
#             self.points = None
#             return

#         # Перевод (y, x) -> (y, x)
#         pts = np.array([(p[0], p[1]) for p in self.path], dtype=np.float32)

#         # --- Удаление дубликатов ---
#         diff = np.linalg.norm(np.diff(pts, axis=0), axis=1)
#         mask = np.insert(diff > 1e-3, 0, True)
#         pts = pts[mask]
#         pts = self._resample_path(pts)

#         # --- Вычисление длины дуги ---
#         distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)
#         arc_lengths = np.insert(np.cumsum(distances), 0, 0.0)

#         self.points = pts
#         self.arc_lengths = arc_lengths
#         self.total_length = arc_lengths[-1]

#         self.current_s = 0.0

#     # Интерполяция пути с равномерной дискретизацией
#     def _resample_path(self, pts: np.ndarray, step: float = 10.0) -> np.ndarray:
#         new_pts = [pts[0]]
#         for i in range(1, len(pts)):
#             p0, p1 = pts[i-1], pts[i]
#             dist = np.linalg.norm(p1 - p0)
#             n = max(int(dist // step), 1)
#             for j in range(1, n+1):
#                 new_pts.append(p0 + (p1 - p0) * j / n)
#         return np.array(new_pts)

#     # ================= Интерполяция =================
#     def _interpolate(self, s: float) -> tuple[np.ndarray, np.ndarray]:
#         """
#         Возвращает:
#             position: np.ndarray (2,)
#             tangent : np.ndarray (2,) (нормализованный)
#         """

#         if self.points is None or self.arc_lengths is None:
#             return np.zeros(2), np.zeros(2)

#         if s >= self.total_length:
#             return self.points[-1], np.zeros(2)

#         # --- поиск сегмента ---
#         idx = np.searchsorted(self.arc_lengths, s) - 1
#         idx = np.clip(idx, 0, len(self.points) - 2)

#         s0 = self.arc_lengths[idx]
#         s1 = self.arc_lengths[idx + 1]

#         t = (s - s0) / (s1 - s0 + 1e-6)

#         p0 = self.points[idx]
#         p1 = self.points[idx + 1]

#         # --- линейная интерполяция ---
#         position = (1 - t) * p0 + t * p1

#         # --- касательный вектор ---
#         tangent = p1 - p0
#         norm = np.linalg.norm(tangent)

#         if norm > 1e-6:
#             tangent = tangent / norm
#         else:
#             tangent = np.zeros(2)

#         return position, tangent

#     # ================= Основная функция =================
#     def get_velocity(self, consumpted_time: float) -> np.ndarray:
#         """
#         Возвращает скорость (vx, vy) с постоянным модулем.

#         Args:
#             consumpted_time: время шага (dt) [s]

#         Returns:
#             np.ndarray shape (2,)
#         """

#         if self.points is None:
#             return np.zeros(2, dtype=np.float32)

#         v: float = self.config.move.max_speed"]

#         # --- обновление положения вдоль сплайна ---
#         self.current_s += v * consumpted_time

#         if self.current_s >= self.total_length:
#             return np.zeros(2, dtype=np.float32)

#         _, tangent = self._interpolate(self.current_s)

#         velocity = v * tangent

#         return velocity.astype(np.float32)
    
#     # Отрисовка сплайна на изображении
#     def draw_spline(
#         self,
#         image: np.ndarray,
#         step: float = 1.0,
#         point_radius: int = 20,
#         line_thickness: int = 5,
#         color_points: tuple = (0, 255, 255),
#         color_line: tuple = (0, 200, 200),
#     ) -> None:
#         """
#         Визуализация сплайна.

#         Args:
#             image: np.ndarray — изображение (BGR)
#             step: float — шаг дискретизации по длине дуги [px]
#             point_radius: int — радиус точек
#             line_thickness: int — толщина линии
#             color_points: tuple — цвет точек
#             color_line: tuple — цвет линии
#         """

#         if self.points is None or self.arc_lengths is None:
#             return

#         if self.total_length <= 1e-6:
#             return

#         # --- дискретизация по длине дуги ---
#         s_values = np.arange(0, self.total_length, step)

#         spline_points: list[tuple[int, int]] = []
        
#         for s in s_values:
#             pos, _ = self._interpolate(s)

#             # (y, x) -> OpenCV (x, y)
#             pt = (int(pos[1]), int(pos[0]))
#             pt: tuple[int, int] = self._grid_to_pixel(pt)
#             spline_points.append(pt)

#         # --- отрисовка точек ---
#         for pt in spline_points:
#             cv2.circle(
#                 image,
#                 pt,
#                 point_radius,
#                 color_points,
#                 -1
#             )
            
#         # --- отрисовка линий ---
#         for i in range(len(spline_points) - 1):
#             cv2.line(
#                 image,
#                 spline_points[i],
#                 spline_points[i + 1],
#                 color_line,
#                 line_thickness
#             )
            
#         pos, _ = self._interpolate(self.current_s)
#         cv2.circle(image, (int(pos[0]), int(pos[1])), 10, (0, 0, 255), -1)
        
#     # Пересчёт положения точки пути на центр дискретизированного пикселя
#     def _grid_to_pixel(
#         self,
#         node: tuple[int, int],
#     ) -> tuple[int, int]:
        
#         return (node[0] * self.grid_step + self.grid_step // 2,
#                 node[1] * self.grid_step + self.grid_step // 2)

class SplineController:
    def __init__(self) -> None:
        self.config = settings

        self.points: np.ndarray | None = None
        self.s: np.ndarray | None = None

        # коэффициенты кубического полинома
        self.ax = self.bx = self.cx = self.dx = None
        self.ay = self.by = self.cy = self.dy = None

        self.current_s: float = 0.0
        self.total_length: float = 0.0

    # ================= Загрузка пути =================
    def load_path(self, path: list[tuple[int, int]]) -> None:
        if len(path) < 2:
            return

        pts = np.array([(p[1], p[0]) for p in path], dtype=np.float32)

        # длина дуги
        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.insert(np.cumsum(ds), 0, 0.0)

        self.points = pts
        self.s = s
        self.total_length = s[-1]

        self._compute_spline()
        self.current_s = 0.0

    # ================= Кубический сплайн =================
    def _compute_spline(self) -> None:
        x = self.points[:, 0]
        y = self.points[:, 1]
        s = self.s

        self.ax, self.bx, self.cx, self.dx = self._cubic_spline_coeffs(s, x)
        self.ay, self.by, self.cy, self.dy = self._cubic_spline_coeffs(s, y)

    def _cubic_spline_coeffs(self, s, values):
        n = len(s) - 1
        h = np.diff(s)

        alpha = np.zeros(n)
        for i in range(1, n):
            alpha[i] = (3/h[i])*(values[i+1]-values[i]) - (3/h[i-1])*(values[i]-values[i-1])

        l = np.ones(n+1)
        mu = np.zeros(n+1)
        z = np.zeros(n+1)

        for i in range(1, n):
            l[i] = 2*(s[i+1]-s[i-1]) - h[i-1]*mu[i-1]
            mu[i] = h[i]/l[i]
            z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]

        c = np.zeros(n+1)
        b = np.zeros(n)
        d = np.zeros(n)
        a = values[:-1]

        for j in reversed(range(n)):
            c[j] = z[j] - mu[j]*c[j+1]
            b[j] = (values[j+1]-values[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
            d[j] = (c[j+1]-c[j])/(3*h[j])

        return a, b, c[:-1], d

    # ================= Поиск сегмента =================
    def _find_segment(self, s_val: float) -> int:
        return np.searchsorted(self.s, s_val) - 1

    # ================= Позиция =================
    def _position(self, s_val: float) -> np.ndarray:
        i = self._find_segment(s_val)
        ds = s_val - self.s[i]

        x = self.ax[i] + self.bx[i]*ds + self.cx[i]*ds**2 + self.dx[i]*ds**3
        y = self.ay[i] + self.by[i]*ds + self.cy[i]*ds**2 + self.dy[i]*ds**3

        return np.array([x, y], dtype=np.float32)

    # ================= Производная =================
    def _derivative(self, s_val: float) -> np.ndarray:
        i = self._find_segment(s_val)
        ds = s_val - self.s[i]

        dx = self.bx[i] + 2*self.cx[i]*ds + 3*self.dx[i]*ds**2
        dy = self.by[i] + 2*self.cy[i]*ds + 3*self.dy[i]*ds**2

        return np.array([dx, dy], dtype=np.float32)

    # ================= Основная функция =================
    def get_velocity(self, dt: float) -> np.ndarray:
        if self.points is None:
            return np.zeros(2, dtype=np.float32)

        v = self.config.move.max_speed

        self.current_s += v * dt

        if self.current_s >= self.total_length:
            return np.zeros(2, dtype=np.float32)

        tangent = self._derivative(self.current_s)
        norm = np.linalg.norm(tangent)

        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)

        direction = tangent / norm
        return (v * direction).astype(np.float32)