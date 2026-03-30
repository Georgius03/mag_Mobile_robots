import numpy as np
import yaml, time
import cv2

from typing import Tuple, List, Dict

from src.utils.classes import Logger, CameraProcessor, RobotinoUnit, AStarPlanner, SplineController

class ControlInterface:
    # ================= Инициализация системы =================
    def __init__(self, config) -> None:
        # Загрузка параметров из файла конфигурации .yaml
        self.config = config
        self.interface_name = "ControlInterface"
        
        # Инициализация компонентов системы
        self.logger = Logger(filename=self.config["utils"]["log_dir"] + "robot_motion.csv")
        self.camera_processor = CameraProcessor(config=self.config)
        self.robot = RobotinoUnit(config=self.config)
        self.grid_planner = AStarPlanner(config=self.config)
        self.spline_controller = SplineController(config=self.config)
        
        # Переменные окружения
        self.click_point: np.ndarray | None = None
        self.motion_started_flag: bool = False
        self.start_time_task: float | None = None
        self.click_point_changed: bool = False
        
        self.path: List[Tuple[int, int]] = []
        self.trajectory: List[np.ndarray] = []
        
        self._setup_capture()
        self._setup_ui()
    
    # Настройка видеопотока
    def _setup_capture(self):
        if self.config['camera']['online']:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Ошибка открытия видео")
        else:
            self.cap = cv2.VideoCapture(self.config['utils']['video_path'])
    
    # Настройка интерфейса и Callback мыши
    def _setup_ui(self):
        cv2.namedWindow(self.interface_name)
        cv2.setMouseCallback(self.interface_name, self._mouse_callback)
        
    # Отклик при нажатии кнопки мыши
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:

        if event == cv2.EVENT_LBUTTONDOWN:
            scale: int = 4
            click_point = np.array((y * scale, x * scale), dtype=np.int32)

            self.motion_started_flag = True
            self.start_time_task = time.time()
            self.trajectory.clear()
            self.click_point_changed = True
            
            self.logger.reset_and_start()
            
            print(f"Целевая точка: {click_point}")

    # Основной цикл работы интерфейса
    def run(self):
        
        try:
            while True:
                start_time: float = time.time()
            
                # ================= ЭТАП 1: Захват видео =================
                ret, frame = self.cap.read()
                
                if not self.config['camera']['online']:
                    D = 1900 - 1080
                    frame = frame[:, D//2:-D//2]
                
                if not ret:
                    break
                
                self.key = cv2.waitKey(1) & 0xFF
                
                if self.key == ord("q"):
                    break
                
                self.camera_processor.load_frame(frame)
                
                # ================= ЭТАП 2: Калбировка системы =================
                if not self.camera_processor.system_calibrated_flag:
                    
                    # Если нажата клавиша "c" — выполняем калибровку по ArUco маркерам
                    if self.key == ord("c"):
                        do_calibrate = True
                    else:
                        do_calibrate = False
                        
                    self.camera_processor.calibrate_system(do_calibrate=do_calibrate)
                        
                    
                # ================= ЭТАП 3: Обработка потока видео =================
                else:
                    
                    # ================= ЭТАП 3.1: Обнаружение препятствий =================
                    robot_position_raw, robot_angular_position = self.camera_processor.get_robot_attitude()  # [x, y], angle
                    self.robot.update_state(curr_pose=robot_position_raw, angle=robot_angular_position)
                    
                    # ================= ЭТАП 3.2: Планирование маршрута =================
                
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

                    v_att = v_att_old + (v_att - v_att_old) * config['move']['filter_gain']

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
                    # Vx, Vy = 0.0, 0.0

                    if motion_started:
                        trajectory.append(robot_position.copy())
                        write_log(current_time - start_time_task, robot_position, np.array([Vx, Vy]), np.array(curr_target))
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
                        step=config['grid']['step'],
                        circle_color=(0, 128, 255),
                        line_color=(0, 128, 255),
                    )

                    # Отрисовка траектории реального движения робота
                    
                    
                    # ================= ЭТАП 3.3: Расчёт сплайна =================
                    
                    
                    # ================= ЭТАП 3.4: Движение по сплайну =================
                    self.robot.navigate_velocity(velocity=(0, 0), omega=0)

                
                print(f"Latency = {time.time() - start_time:.4f} s ")

                display = self.camera_processor.get_display()
                
                # ================= ЭТАП 4: Визуализация =================
                if len(self.trajectory) > 1:
                    for i in range(1, len(self.trajectory)):
                        pt1 = tuple(self.trajectory[i-1][::-1].astype(int))
                        pt2 = tuple(self.trajectory[i][::-1].astype(int))
                        cv2.line(display, pt1, pt2, (255, 0, 255), 10)
                
                display: np.ndarray = cv2.resize(
                    display,
                    (self.config["map_params"]["resolution"] // 4, self.config["map_params"]["resolution"] // 4)
                )
                cv2.imshow(self.interface_name, display)




        # Обработка отключения клиента
        except (ConnectionResetError, BrokenPipeError):
            print("Client disconnected.")
            
        # Освобождение ресурсов
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            
            if self.logger is not None:
                self.logger.close()
                
            if self.config["socket_params"]["enable"]:
                self.robot.disconnect()
            print("Final")
    

# Точка входа
def main():
    # Загрузка параметров из файла конфигурации .yaml
    with open("parameters.yaml") as config_file:
        config = yaml.safe_load(config_file)
    print(f"System Config: {config}")
    
    # Инициализация и запуск интерфейса управления
    interface = ControlInterface(config)
    interface.run()

# Запуск программы
if __name__ == "__main__":
    main()