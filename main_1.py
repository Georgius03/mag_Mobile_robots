import math, time, csv
import socket, requests
import yaml

import cv2
import cv2.aruco as aruco
import numpy as np

from typing import Tuple, Optional

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
    k_p: float = 1.0
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

    if event == cv2.EVENT_LBUTTONDOWN:
        scale: int = 4
        click_point = np.array((y * scale, x * scale), dtype=np.int32)
        print("click_point id:", id(click_point), " value:", click_point, 'in function')
        motion_started = True
        start_time_task = time.time()
        trajectory.clear()
        current_target_index = 0


        print(f"Целевая точка: {click_point}")

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
            print(f"Sent Vx: {vx}, Vy: {vy}, w: {omega}")
        else:
            print(f"Send error: {response.status_code} - {response.text}")
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


def main():

    global click_point
    global motion_started
    global motion_started_flag
    global start_time_task
    global trajectory
    global log_file

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
        cap = cv2.VideoCapture(config['camera']['video_dir'])

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

    perspective_matrix: Optional[np.ndarray] = None
    calibrated: bool = False

    cv2.namedWindow("map_planner")
    cv2.setMouseCallback('map_planner', mouse_callback)

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

            # --- Проекция ---
            mask = project_mask(
                mask,
                H=config['camera']['H'],
                h=config['camera']['h'],
                center=trans_center
            )

            WALL_WIDTH = config['apf']['WALL_WIDTH']
            mask[:WALL_WIDTH, :] = 255
            mask[-WALL_WIDTH:, :] = 255
            mask[:, :WALL_WIDTH] = 255
            mask[:, -WALL_WIDTH:] = 255
            # mask[0:-1, 1000:1700] = 255

            
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(working_area, contours, -1, (0, 255, 0), 2)

            # ================= УПРАВЛЕНИЕ ДВИЖЕНИЕМ =================

            if ids is not None:

                current_time: float = time.time()

                if motion_started and not motion_started_flag:
                    motion_started_flag = True
                    init_logger("robot_motion_log_1.csv")
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
                
                # Отрисовка точки положения робота
                cv2.circle(working_area, robot_position[::-1], 25, (0, 0, 255), -1)

                if click_point is None:
                    click_point = robot_position.copy()

                v_att, dist = compute_velocity(
                    current_position=robot_position,
                    target_position=click_point,
                    v_max=config['move']['max_speed'],
                    tolerance=config['move']['dist_stop'],
                    k_p=config['move']['k_prop']
                )

                v_rep = compute_repulsive_field(
                     robot_position=robot_position,
                     obstacle_mask=mask,
                     # obstacle_mask=mask,
                     d0=config['apf']['d0'],
                     k_rep=config['apf']['k_rep'],
                     influence_radius=config['apf']['influence_radius'],
                     v_max=config['move']['max_speed'],
                     scale_factor=config['apf']['scale_factor']
                )


                velocity: np.ndarray = v_att + v_rep
                speed: float = float(np.linalg.norm(velocity))

                print(dist)
                if dist:
                    velocity *= 0
                    motion_started = False
                print(velocity)

                if speed > config['move']['max_speed'] and speed > 1e-6:
                    velocity = velocity / speed * config['move']['max_speed']


                print(velocity)
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
                        pass
                else:
                    if config['socket_params']['enable']:
                        send_velocity(0, 0, 0)
                        pass
                    pass
                

                # Отрисовка направления
                cv2.line(working_area, robot_position[::-1].astype(int), click_point[::-1].astype(int), (255, 0, 255), 3)

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
                    
                cv2.circle(working_area, (click_point[1], click_point[0]), 25, (0, 255, 0), -1)
                if len(trajectory):
                    cv2.circle(working_area, trajectory[0][::-1], 25, (0, 128, 255), -1)

                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        pt1 = tuple(trajectory[i-1][::-1].astype(int))
                        pt2 = tuple(trajectory[i][::-1].astype(int))
                        cv2.line(working_area, pt1, pt2, (0, 255, 255), 5)
                    
                print(f"Latency = {time.time() - start_time:.4f} s ||| Vx = {Vx:.4f} px/s | Vy = {Vy:.4f} px/s | angle = {math.degrees(angle)}")
                
            # -------- Отображение --------
            display: np.ndarray = cv2.resize(
                working_area,
                (OUTPUT_SIZE[0] // 4, OUTPUT_SIZE[1] // 4)
            )

            cv2.imshow("map_planner", display)

            if key == ord("q"):
                break

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