import math, time, csv
import yaml
import heapq

import cv2
import cv2.aruco as aruco
import numpy as np

from typing import Tuple, Optional, List, Dict

from robotino_communication import *
from apf_utils import *


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
        cap = cv2.VideoCapture(config['camera']['video_path'])

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
                    init_logger(config['utils']['log_dir'] + "robot_motion_log_2_3.csv")
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