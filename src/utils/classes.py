import math, csv, socket, requests
import heapq

import cv2
import cv2.aruco as aruco
import numpy as np

from src.core.config import settings

from scipy.interpolate import make_interp_spline, make_smoothing_spline

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
                                    "speed_px_s"])
            # Сбрасываем буфер на диск, чтобы заголовок записался сразу
            self.log_file.flush()

    def write_log(self, t: float, pos: tuple[int, int], vel: np.ndarray) -> None:
        speed: float = float(np.linalg.norm(vel))
        if self.csv_writer is not None:
            self.csv_writer.writerow([
                f"{t:.6f}",
                f"{pos[0]:.3f}", f"{pos[1]:.3f}",
                f"{vel[0]:.6f}", f"{vel[1]:.6f}",
                f"{speed:.6f}",
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

        self.ArUco_marker_rotate_matrix = np.array([
            [math.cos(self.config.move.ArUco_angle), -math.sin(self.config.move.ArUco_angle)],
            [math.sin(self.config.move.ArUco_angle), math.cos(self.config.move.ArUco_angle)]
        ])
        
        self.base_rotate_matrix = np.array([
            [math.cos(math.radians(90)), -math.sin(math.radians(90))],
            [math.sin(math.radians(90)), math.cos(math.radians(90))]
        ])
        
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
    def get_robot_attitude(self) -> tuple[tuple, float] | None:
        corners, ids, _ = self.aruco_6x6.detectMarkers(self.frame)

        center_x, center_y, angle = 0, 0, 0
        
        if ids is not None:
            c: np.ndarray = corners[0][0][:, ::-1]

            center_x: int = int(np.mean(c[:, 0]))
            center_y: int = int(np.mean(c[:, 1]))

            dx: float = c[1][0] - c[0][0]
            dy: float = c[1][1] - c[0][1]

            direction_vector = np.array([dx, dy])
            direction_vector = self.base_rotate_matrix @ self.ArUco_marker_rotate_matrix @ direction_vector
            angle = math.atan2(direction_vector[1], direction_vector[0])
        
            pose_2D: np.ndarray = np.array([center_x, center_y], dtype=int)
            pose_2D = self._project_floor_point(
                point=pose_2D,
                H=self.config.camera.H,
                h=self.config.camera.h,
                center=self.trans_center
            )
        
            return tuple(pose_2D), angle
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
        
        self.curr_pose2D: tuple[int, int] = (0, 0)
        self.curr_angle: float = 0.0
        
        self.velocity: np.ndarray = np.array([0.0, 0.0])
        
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
        
        print(f"[INFO] Vel:{velocity}")
        self._send_velocity(velocity[0], velocity[1], 0)
    
    # Управление по скоростям
    def navigate_velocity(self, velocity, omega=0) -> None:
        velocity = self._rotate_vector(velocity)
        self.velocity = velocity
        
        self._send_velocity(velocity[0], velocity[1], omega)
    
    # Обновление текущего положения робота
    def update_state(self, curr_pose: tuple[int, int], angle: float) -> None:
        self.curr_pose2D = curr_pose
        self.curr_angle = angle

    # Поворот вектора скорости с учётом ориентации полигона и поворота робота
    def _rotate_vector(self, vector: np.ndarray) -> np.ndarray:
        rotation_matrix = np.array([[math.cos(self.curr_angle), -math.sin(self.curr_angle)],
                                     [math.sin(self.curr_angle), math.cos(self.curr_angle)]])
        return rotation_matrix @ vector
    
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
        current_position: tuple[int, int],
        target_position: tuple[int, int],
        v_max: float,
        tolerance: float,
        k_p: float = 1
    ) -> tuple[np.ndarray, bool]:
        """
        Вычисляет вектор линейной скорости движения к целевой точке
        в декартовой системе координат.

        Args:
            current_position : tuple[int, int] – текущая позиция [px]
            target_position  : tuple[int, int] – целевая позиция [px]
            v_max            : float – максимальная скорость [px/s]
            tolerance        : float – радиус достижения цели [px]
            k_p              : float – коэффициент пропорционального регулятора [1/s]

        Returns:
            velocity : np.ndarray shape (2,) – вектор скорости [px/s]
            reached  : bool – флаг достижения цели
        """
        current_pos: np.ndarray = np.array(current_position, dtype=np.float32)
        target_pos: np.ndarray = np.array(target_position, dtype=np.float32)
        # --- Вектор ошибки ---
        error: np.ndarray = target_pos - current_pos

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
        path: list[tuple[int, int]],
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

        if len(path) < 2:
            return

        pixel_points: list[tuple[int, int]] = []

        # --- преобразование grid → pixel ---
        for node in path:
            y, x = self._grid_to_pixel(node)
            pixel_points.append((x, y))

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
    
    def path_to_pixels(
        self,
        path: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        path_pixels: list[tuple[int, int]] = []
        for x, y in path:
            path_pixels.append((y * self.grid_step + self.grid_step // 2,
                                x * self.grid_step + self.grid_step // 2))
        return path_pixels


class SplineTrajectoryController:
    def __init__(self, points: list[tuple[int, int]], v_max_meters: float = 0.08, num_samples: int = 200):
        """
        Класс для управления движением по сплайну с фиксированной скоростью V_max.
        """
        self.config = settings
        self.v_max = v_max_meters * 1000
        pts = np.array(points, dtype=float)
        
        # 1. Базовая параметризация по хордам для построения геометрии
        deltas = np.diff(pts, axis=0)
        distances = np.sqrt(np.sum(deltas**2, axis=1))
        
        u_coarse = np.zeros(len(pts))
        u_coarse[1:] = np.cumsum(distances)
        self.total_length = u_coarse[-1]
        
        if self.total_length > 0:
            u_coarse /= self.total_length
            
        # Строим геометрический сплайн
        # self.spline = make_interp_spline(u_coarse, pts, k=3)
        self.spline = make_smoothing_spline(u_coarse, pts, lam=self.config.move.spline_smoothing_lambda)
        
        # 2. Репараметризация: строим точную таблицу "Длина дуги -> Параметр u"
        self.u_samples = np.linspace(0, 1, num_samples)
        interp_points = self.spline(self.u_samples)
        
        # Вычисляем кумулятивную длину дуги для каждого u_sample
        diffs = np.diff(interp_points, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        
        self.arc_lengths = np.zeros(num_samples)
        self.arc_lengths[1:] = np.cumsum(seg_lengths)
        
        # Текущее состояние движения (пройденный путь в единицах координат)
        self.current_distance = 0.0
        self.is_finished = False

    def get_full_path(self, steps: int = 300) -> list[tuple[int, int]]:
        """
        Генерирует список экранных (X, Y) координат всего сплайна для отрисовки линии траектории.
        """
        u_vals = np.linspace(0, 1, steps)
        full_pts = self.spline(u_vals)
        # Переводим в экранные координаты (деление на SCALE и инверсия Y, X в X, Y для OpenCV)
        return [(int(p[1]), int(p[0])) for p in full_pts]

    def update(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Принимает прошедшее время dt и возвращает (position, velocity)
        """
        if self.is_finished:
            end_pos = self.spline(1.0)
            return end_pos, np.zeros(2)
        
        self.current_distance += self.v_max * dt
        
        if self.current_distance >= self.total_length:
            self.current_distance = self.total_length
            self.is_finished = True
            
        current_u = np.interp(self.current_distance, self.arc_lengths, self.u_samples)
        
        pos = self.spline(current_u)
        geom_vel = self.spline(current_u, nu=1)
        
        geom_speed = np.linalg.norm(geom_vel)
        if geom_speed > 0 and not self.is_finished:
            velocity = (geom_vel / geom_speed) * self.v_max
        else:
            velocity = np.zeros(2)
            
        return pos, velocity[::-1] / 1e3