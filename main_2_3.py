import math, time, csv
import socket, requests
import yaml
import heapq

import cv2
import cv2.aruco as aruco
import numpy as np

from typing import Tuple, Optional, List, Dict

with open('parameters.yaml') as config_file:
    config = yaml.safe_load(config_file)
print(config)

click_point: np.ndarray | None = None
motion_started_flag: bool = False
motion_started: bool = False
start_time_task: float | None = None
trajectory: list[np.ndarray] = []
log_file = None
csv_writer: csv.writer = None
current_target_index: int = 0
need_replan: bool = False
click_point_changed: bool = False
path: List[Tuple[int, int]] = []
enable_replan: bool = False


# Расчёт вектора скорости от потенциального поля
def compute_repulsive_velocity_fast(
    robot_position: np.ndarray,
    dist: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    d0: float,
    k_rep: float,
    influence_radius: int
) -> np.ndarray:
    """
    Рассчитывает результирующий отталкивающий вектор потенциального поля
    вокруг робота на основе уменьшенной маски.

    Args:
        robot_position: np.ndarray shape (2,) – координаты робота на уменьшенной маске
        dist: np.ndarray – distance transform уменьшенной маски
        grad_x: np.ndarray – нормализованный градиент по X
        grad_y: np.ndarray – нормализованный градиент по Y
        d0: радиус влияния [px] (уменьшенный)
        k_rep: коэффициент отталкивания
        influence_radius: радиус поиска препятствий вокруг робота [px] (уменьшенный)

    Returns:
        np.ndarray shape (2,) – результирующий вектор поля [px/s]
    """
    cx, cy = int(robot_position[0]), int(robot_position[1])
    h, w = dist.shape

    if cx <= 1 or cy <= 1 or cx >= w-2 or cy >= h-2:
        return np.zeros(2, dtype=np.float32)

    R = influence_radius
    x_min = max(cx - R, 1)
    x_max = min(cx + R, w - 2)
    y_min = max(cy - R, 1)
    y_max = min(cy + R, h - 2)

    local_dist = dist[y_min:y_max, x_min:x_max].astype(np.float32)
    local_gx = grad_x[y_min:y_max, x_min:x_max]
    local_gy = grad_y[y_min:y_max, x_min:x_max]

    yy, xx = np.meshgrid(
        np.arange(y_min, y_max),
        np.arange(x_min, x_max),
        indexing="ij"
    )

    dx = xx - cx
    dy = yy - cy
    circle_mask = (dx*dx + dy*dy) <= R*R

    eps = 1e-3
    local_dist = np.maximum(local_dist, eps)
    valid = (local_dist < d0) & circle_mask

    if not np.any(valid):
        return np.zeros(2, dtype=np.float32)

    magnitude = np.zeros_like(local_dist, dtype=np.float32)
    magnitude[valid] = k_rep * (1.0/local_dist[valid] - 1.0/d0) / (local_dist[valid]**2)

    fx = float(np.sum(magnitude * local_gx))
    fy = float(np.sum(magnitude * local_gy))

    force = np.array([fx, fy], dtype=np.float32)
    if not np.all(np.isfinite(force)):
        return np.zeros(2, dtype=np.float32)

    return force

# Расчёт вектора скорости от потенциального поля
def compute_repulsive_field(
    robot_position: np.ndarray,
    obstacle_mask: np.ndarray,
    d0: float,
    k_rep: float,
    influence_radius: int,
    v_max: float,
    scale_factor: float = 0.5
) -> np.ndarray:
    """
    Рассчитывает результирующий вектор отталкивающего поля вокруг робота.
    Маска автоматически уменьшается для ускорения расчёта.

    Args:
        robot_position: np.ndarray shape (2,) – координаты робота [px]
        obstacle_mask: np.ndarray – бинарная маска препятствий
        d0: радиус влияния препятствия [px]
        k_rep: коэффициент отталкивания
        influence_radius: радиус вычисления поля вокруг робота [px]
        scale_factor: коэффициент уменьшения изображения для ускорения

    Returns:
        v_rep: np.ndarray shape (2,) – результирующий вектор поля [px/s]
    """
    # --- Масштабируем маску и координаты робота ---
    obstacle_mask = cv2.transpose(obstacle_mask)
    small_mask = cv2.resize(
        obstacle_mask,
        (int(obstacle_mask.shape[1]*scale_factor), int(obstacle_mask.shape[0]*scale_factor)),
        interpolation=cv2.INTER_NEAREST
    )
    small_robot_pos = robot_position * scale_factor
    small_d0 = d0 * scale_factor
    small_influence_radius = int(influence_radius * scale_factor)

    # --- Distance Transform ---
    dist = cv2.distanceTransform(255 - small_mask, cv2.DIST_L2, 3)

    # --- Градиенты и нормализация ---
    grad_y, grad_x = np.gradient(dist.astype(np.float32))  # axis0 = y, axis1 = x
    norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
    grad_x /= norm
    grad_y /= norm

    # --- Расчёт отталкивающей силы ---
    v_rep = compute_repulsive_velocity_fast(
        robot_position=small_robot_pos,
        dist=dist,
        grad_x=grad_x,
        grad_y=grad_y,
        d0=small_d0,
        k_rep=k_rep,
        influence_radius=small_influence_radius
    )

    # --- Масштабируем обратно ---
    v_rep /= scale_factor * 1000

    speed: float = float(np.linalg.norm(v_rep))

    # --- ограничение максимальной скорости ---
    if speed > v_max and speed > 1e-6:
        v_rep = v_rep / speed * v_max

    return v_rep

# Расчёт результирующего вектора скорости робота
def compute_velocity(
    current_position: np.ndarray,
    target_position: np.ndarray,
    v_max: float,
    tolerance: float,
    k_p: float = 1
) -> Tuple[np.ndarray, bool]:
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

# Расчёт матрицы преобразования перспективы
def compute_perspective(
        corners: np.ndarray,
        ids: np.ndarray,
        dst_points: np.ndarray,
        output_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Выполняет калибровку полигона по 4 ArUco маркерам.
    """
    markers: dict = {}

    for i, corner in enumerate(corners):
        marker_id: int = int(ids[i][0])
        markers[marker_id] = corner[0]

    src_points: np.ndarray = np.float32([
        markers[config['aruco']['left_up']][0],
        markers[config['aruco']['right_up']][1],
        markers[config['aruco']['right_down']][2],
        markers[config['aruco']['left_down']][3]
    ])

    matrix: np.ndarray = cv2.getPerspectiveTransform(src_points, dst_points)

    return matrix, src_points

# Проецирование точки на поверхность с учётом камеры
def project_point(
    point: np.ndarray,
    H: float,
    h: float,
    center: Tuple[float, float]
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

# Расчёт положения спроецированного исходного центра изображения (для коррекции маски)
def compute_projected_center(
    perspective_matrix: np.ndarray,
    frame_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Возвращает положение оптического центра камеры
    после применения warpPerspective.
    """

    height: int = frame_shape[0]
    width: int = frame_shape[1]

    original_center: np.ndarray = np.array(
        [[[width / 2.0, height / 2.0]]],
        dtype=np.float32
    )

    warped_center: np.ndarray = cv2.perspectiveTransform(
        original_center,
        perspective_matrix
    )

    return warped_center[0][0]

# Проекция маски на поверхность с учётом камеры
def project_mask(
    mask: np.ndarray,
    H: float,
    h: float,
    center: Tuple[float, float]
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

# Отклик при нажатии кнопки cccмыши
def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
    global click_point
    global motion_started
    global start_time_task
    global trajectory
    global current_target_index
    global click_point_changed
    global enable_replan

    if event == cv2.EVENT_LBUTTONDOWN:
        scale: int = 4
        click_point = np.array((y * scale, x * scale), dtype=np.int32)

        motion_started = True
        start_time_task = time.time()
        trajectory.clear()
        current_target_index = 0
        click_point_changed = True
        
        print(f"Целевая точка: {click_point}")
    
    if event == cv2.EVENT_RBUTTONDOWN:
        enable_replan = not enable_replan
        if enable_replan:
            print(f"Replan Enabled!")
        else:
            print("Replan Disabled!")

# Подключение к Robotino
def connect_to_robotino():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((config['socket_params']['IP_ADDRESS'], config['socket_params']['PORT']))
        print("Successfully connected")
        return sock
    except Exception as e:
        print(f"Error connecting: {e}")
        return None

# Отправка управляющих сигналов -
def send_velocity(vx:float, vy:float, omega:float):
    url = f"http://{config['socket_params']['IP_ADDRESS']}/data/omnidrive"
    data = [vx, vy, omega]
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            # print(f"Sent Vx: {vx}, Vy: {vy}, w: {omega}")
            pass
        else:
            print(f"Send error: {response.status_code} - {response.text}")
            pass
    except Exception as e:
        print(f"Error sending data: {e}")

# Инициализация файла для логирования
def init_logger(filename: str):
    global log_file, csv_writer
    log_file = open(filename, "w", newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["time_s", "pos_x_px", "pos_y_px", "vx_px_s", "vy_px_s", "speed_px_s"])

# Логирование данных в файл
def write_log(t: float, pos: np.ndarray, vel: np.ndarray):
    global csv_writer
    speed: float = float(np.linalg.norm(vel))
    csv_writer.writerow([
        f"{t:.6f}",
        f"{pos[0]:.3f}",
        f"{pos[1]:.3f}",
        f"{vel[0]:.6f}",
        f"{vel[1]:.6f}",
        f"{speed:.6f}"
    ])

# Пулинг маски для уменьшения дискретизации
def maxpool2D(
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
            1 = занято
            0 = свободно
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

# Расчёт расстояния между точками старта и целью
def heuristic(
    a: Tuple[int, int],
    b: Tuple[int, int]
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
def reconstruct_path(
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
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

    path: List[Tuple[int, int]] = [goal]
    current: Tuple[int, int] = goal

    # Двигаемся назад по дереву поиска
    while current != start:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path

# Алгоритм A*
def astar(
    pooled_mask: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
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

    if pooled_mask[goal] == 0:
        return []

    # --- Очередь с приоритетом ---
    # Храним (f_cost, координата)
    open_set: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_set, (0.0, start))

    # Словарь предков для восстановления пути
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # g_cost хранит реальную стоимость пути от старта
    g_cost: Dict[Tuple[int, int], float] = {
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
            return reconstruct_path(came_from, start, goal)

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
            neighbor: Tuple[int, int] = (ny, nx)

            # Если путь к соседу найден впервые
            # или найден более короткий путь
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:

                g_cost[neighbor] = tentative_g
                f_cost: float = tentative_g + heuristic(neighbor, goal)

                heapq.heappush(open_set, (f_cost, neighbor))
                came_from[neighbor] = current

    # Если очередь пуста — путь не существует
    return []

# Пересчёт положения точки пути на центр дискретизированного пикселя
def grid_to_pixel(
    node: Tuple[int, int],
    grid_step: int
) -> Tuple[int, int]:

    return np.array((node[0] * grid_step + grid_step // 2,
                     node[1] * grid_step + grid_step // 2))

# Визуализация пути
def draw_path(
    image: np.ndarray,
    path: list[tuple[int, int]],
    step: int,
    point_radius: int = 20,
    line_thickness: int = 5
) -> None:
    """
    Отрисовывает путь A* поверх изображения:
    - узлы (круги)
    - соединяющие сегменты (линии)

    :param image: рабочее изображение (BGR)
    :param path: список grid-узлов [(row, col), ...]
    :param step: размер клетки сетки (px)
    :param point_radius: радиус отображаемых точек (px)
    :param line_thickness: толщина линии (px)
    """

    if len(path) < 2:
        return

    pixel_points: list[tuple[int, int]] = []

    # --- преобразование grid → pixel ---
    for node in path:
        pixel_point: tuple[int, int] = grid_to_pixel(node, step)
        # OpenCV: (x, y)
        pixel_points.append((pixel_point[1], pixel_point[0]))

    # --- отрисовка линий ---
    for i in range(len(pixel_points) - 1):
        cv2.line(
            image,
            pixel_points[i],
            pixel_points[i + 1],
            (0, 128, 255),
            line_thickness
        )

    # --- отрисовка узлов ---
    for point in pixel_points:
        cv2.circle(
            image,
            point,
            point_radius,
            (128, 0, 255),
            -1
        )


def main():
    global click_point
    global motion_started
    global motion_started_flag
    global start_time_task
    global trajectory
    global log_file
    global current_target_index
    global need_replan
    global click_point_changed
    global path
    global enable_replan

    if config['socket_params']['enable']:
        sock = connect_to_robotino()

    OUTPUT_SIZE: Tuple[int, int] = (config['map_params']['resolution'], config['map_params']['resolution'])
    DST_POINTS: np.ndarray = np.float32([
        [0, 0],
        [config['map_params']['resolution'], 0],
        [config['map_params']['resolution'], config['map_params']['resolution']],
        [0, config['map_params']['resolution']]
    ])
    
    if config['camera']['sharpening']:
        SHARPENING_KERNEL = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

    if config['camera']['online']:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(r"computer_vision_mag\images\2026-02-1713-42-22.mp4")

    if not cap.isOpened():
        raise RuntimeError("Ошибка открытия видео")

    aruco_5x5 = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_5X5_100),
        aruco.DetectorParameters()
    )

    aruco_6x6 = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_6X6_100),
        aruco.DetectorParameters()
    )

    kernel_denoise: np.ndarray = np.ones((10, 10), np.uint8)
    robot_radius_dilatation_pixels: int = config['grid']['robot_radius']
    
    perspective_matrix: Optional[np.ndarray] = None
    calibrated: bool = False

    cv2.namedWindow("map_planner")
    cv2.setMouseCallback('map_planner', mouse_callback)

    v_att_old = np.array([0, 0])

    try:
        while True:
            start_time: float = time.time()

            ret, frame = cap.read()
            if config['camera']['sharpening']:
                frame = cv2.filter2D(frame, -1, SHARPENING_KERNEL)

            if not ret:
                break
            
            if not calibrated:
                corners, ids, _ = aruco_5x5.detectMarkers(frame)
                aruco.drawDetectedMarkers(frame, corners, ids)

            key: int = cv2.waitKey(1)
            # ================= ЭТАП I: КАЛИБРОВКА =================
            if key == ord("c"):
                try:
                    if ids is None or len(ids) < 4:
                        raise ValueError("Недостаточно маркеров для калибровки")
                    
                    perspective_matrix, _ = compute_perspective(
                        corners,
                        ids,
                        DST_POINTS,
                        OUTPUT_SIZE
                    )

                    original_center_in_area = compute_projected_center(
                        perspective_matrix=perspective_matrix,
                        frame_shape=frame.shape[:2]
                    )

                    trans_center = (original_center_in_area[1], original_center_in_area[0])

                    calibrated = True
                    print("Калибровка выполнена")

                except ValueError as e:
                    print(e)

            if key == ord("q"):
                break

            if not calibrated:
                # frame = cv2.resize(
                #     frame,
                #     (frame.shape[1] // 2, frame.shape[0] // 2)
                # )
                cv2.imshow("map_planner", frame)
                continue

            # ================= ЭТАП II: ОСНОВНАЯ ОБРАБОТКА =================

            working_area: np.ndarray = cv2.warpPerspective(
                frame,
                perspective_matrix,
                OUTPUT_SIZE
            )

            rgb_frame: np.ndarray = working_area.copy()

            # -------- Трекинг робота --------
            corners, ids, _ = aruco_6x6.detectMarkers(rgb_frame)

            if ids is not None:
                c: np.ndarray = corners[0][0]

                center_x: int = int(np.mean(c[:, 1]))
                center_y: int = int(np.mean(c[:, 0]))

                # print(center_x, center_y)

                dx: float = c[1][0] - c[0][0]
                dy: float = c[1][1] - c[0][1]

                angle: float = math.atan2(dx, dy)

            # --- HSV ---
            hsv = cv2.cvtColor(working_area, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, (0, 100, 135), (179, 255, 255))

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_denoise)
            mask = cv2.erode(mask, kernel=kernel_denoise, iterations=2)
            mask = cv2.dilate(mask, kernel=kernel_denoise, iterations=2)

            inv_mask = cv2.bitwise_not(mask)
            dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
            expanded = (dist <= robot_radius_dilatation_pixels).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, expanded)

            # --- Проекция ---
            mask = project_mask(
                mask,
                H=config['camera']['H'],
                h=config['camera']['h'],
                center=trans_center
            )
            
            WALL_WIDTH = config['apf']['WALL_WIDTH2']
            mask[:WALL_WIDTH, :] = 255
            mask[-WALL_WIDTH:, :] = 255
            mask[:, :WALL_WIDTH] = 255
            mask[:, -WALL_WIDTH:] = 255

            
            pooled_mask = maxpool2D(inflated_mask=mask,
                                      grid_step=config['grid']['step'])
            
            visualize_pool_mask = cv2.resize(pooled_mask, (config['map_params']['resolution'], config['map_params']['resolution']), interpolation=cv2.INTER_NEAREST)

            pooled_contours, _ = cv2.findContours(
                visualize_pool_mask,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(working_area, pooled_contours, -1, (255, 0, 0), 2)
            cv2.drawContours(working_area, contours, -1, (0, 255, 0), 2)

            # ================= УПРАВЛЕНИЕ ДВИЖЕНИЕМ =================

            if ids is not None:

                current_time: float = time.time()

                if motion_started and not motion_started_flag:
                    motion_started_flag = True
                    init_logger("robot_motion_log_2_3.csv")
                    print("Движение начато")
                elif not motion_started and motion_started_flag:
                    motion_started_flag = False
                
                robot_position_raw: np.ndarray = np.array(
                    (center_x, center_y),
                    dtype=np.float32
                )

                robot_position: np.ndarray = project_point(
                    point=robot_position_raw,
                    H=config['camera']['H'],
                    h=config['camera']['h'],
                    center=trans_center
                )

                if click_point is None:
                    click_point = robot_position.copy()

                start_node: Tuple[int, int] = (robot_position[0] // config['grid']['step'],
                                               robot_position[1] // config['grid']['step'])
                
                goal_node: Tuple[int, int] = (click_point[0] // config['grid']['step'],
                                              click_point[1] // config['grid']['step'])

                if motion_started and len(path) == 0:
                    need_replan = True

                if click_point_changed:
                    click_point_changed = False
                    need_replan = True

                if need_replan or enable_replan:
                    path = astar(
                        pooled_mask,
                        start_node,
                        goal_node
                        )
                    need_replan = False

                if len(path) > 0:
                    if current_target_index >= len(path):
                        current_target_index = len(path) - 1

                    curr_node = path[current_target_index]
                    curr_target = grid_to_pixel(curr_node, config['grid']['step'])
                    curr_target = curr_target
                else:
                    curr_target = robot_position.copy()

                v_att, dist = compute_velocity(
                    current_position=robot_position,
                    target_position=curr_target,
                    v_max=config['move']['max_speed'],
                    tolerance=config['move']['dist_stop'],
                    k_p=config['move']['k_prop']
                )

                if enable_replan:
                    v_rep = compute_repulsive_field(
                        robot_position=robot_position,
                        obstacle_mask=mask,
                        d0=55,
                        k_rep=2.0e-10,
                        influence_radius=200,
                        v_max=config['move']['max_speed'],
                        scale_factor=config['apf']['scale_factor']
                    )
                else:
                    v_rep = np.array([0, 0])

                v_att = v_att_old + (v_att - v_att_old) * 0.2

                v_att_old = v_att
                velocity: np.ndarray = v_att + v_rep
                speed: float = float(np.linalg.norm(velocity))

                if speed > config['move']['max_speed'] and speed > 1e-6:
                    velocity = velocity / speed * config['move']['max_speed']

                if dist:
                    current_target_index += 1
                    if current_target_index >= len(path):
                        current_target_index = len(path)
                    print(f"Point {current_target_index} reached! Moving to {current_target_index + 1}")

                if float(np.linalg.norm(robot_position - click_point)) <= 30:
                    motion_started = False

                comp_matrix = np.array([[0, -1],
                                        [1, 0]])

                rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                            [math.sin(angle), math.cos(angle)]])

                Vx, Vy = velocity @ comp_matrix @ rotation_matrix

                if motion_started:
                    trajectory.append(robot_position.copy())
                    write_log(current_time - start_time_task, robot_position, np.array([Vx, Vy]))
                    if config['socket_params']['enable']:
                        send_velocity(Vx, Vy, 0)
                else:
                    if config['socket_params']['enable']:
                        send_velocity(0, 0, 0)
                    pass
                
                # ========= Визуализация ==========

                # Отрисовка точки положения робота
                cv2.circle(working_area, robot_position[::-1], 25, (0, 0, 255), -1)

                # Отрисовка направления
                cv2.line(working_area, robot_position[::-1], click_point[::-1], (255, 0, 255), 3)

                # Отрисовка вектора скорости
                colors: dict = {
                    0: (0, 0, 255),
                    1: (0, 255, 0),
                    2: (255, 0, 0)
                }
                scale_arrows: dict = {
                    0: 1e3,
                    1: 1e3,
                    2: 1e3,
                }

                # print([v_att, v_rep, velocity])
                for ind, (comp_x, comp_y) in enumerate([v_att, v_rep, velocity]):
                    rep_end = (
                        int(robot_position[1] + comp_y * scale_arrows[ind]),
                        int(robot_position[0] + comp_x * scale_arrows[ind])
                    )

                    cv2.arrowedLine(
                        working_area,
                        tuple(robot_position[::-1].astype(int)),
                        rep_end,
                        colors[ind],
                        15
                    )

                # Отрисовка целевой точки
                cv2.circle(working_area, (click_point[1], click_point[0]), 25, (30, 255, 0), -1)
                if len(trajectory):
                    cv2.circle(working_area, trajectory[0][::-1], 25, (0, 128, 255), -1)

                draw_path(
                    image=working_area,
                    path=path,
                    step=config['grid']['step']
                )

                # Отрисовка траектории реального движения робота
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        pt1 = tuple(trajectory[i-1][::-1].astype(int))
                        pt2 = tuple(trajectory[i][::-1].astype(int))
                        cv2.line(working_area, pt1, pt2, (0, 128, 255), 5)
                    
                print(f"Latency = {time.time() - start_time:.4f} s ||| Vx = {Vx:.4f} px/s | Vy = {Vy:.4f} px/s | angle = {math.degrees(angle)}")
                
            # -------- Отображение --------
            display: np.ndarray = cv2.resize(
                working_area,
                (OUTPUT_SIZE[0] // 4, OUTPUT_SIZE[1] // 4)
            )

            if key == ord("q"):
                break

            cv2.imshow("map_planner", display)

        cap.release()
        cv2.destroyAllWindows()

    except (ConnectionResetError, BrokenPipeError):
        print("Client disconnected.")
    finally:
        if log_file is not None:
            log_file.close()
            
        if config['socket_params']['enable']:
            sock.close()
        print("Final")


if __name__ == '__main__':
    main()