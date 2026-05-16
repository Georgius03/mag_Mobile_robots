import numpy as np
import time
import cv2
import math

from src.utils.classes import Logger, CameraProcessor, RobotinoUnit, AStarPlanner, SplineTrajectoryController
from src.core.config import settings

class ControlInterface:
    # ================= Инициализация системы =================
    def __init__(self) -> None:
        # Загрузка параметров из файла конфигурации .yaml
        self.config = settings
        self.interface_name = "ControlInterface"
        
        # Инициализация компонентов системы
        self.logger = Logger(filename=self.config.utils.log_dir + "robot_motion.csv")
        self.camera_processor = CameraProcessor()
        self.robot = RobotinoUnit()
        self.grid_planner = AStarPlanner()
        self.spline_controller = None
        
        # Переменные окружения
        self.click_point: np.ndarray | None = None
        self.motion_started_flag: bool = False
        self.motion_started: bool = False
        self.start_time_task: float = 0.0
        self.need_replan: bool = False
        
        self.prev_time = time.time()
        
        self.path: list[tuple[int, int]] = []
        self.spline_points: list[tuple[int, int]] = []
        self.trajectory: list[tuple[int, int]] = []
        self.calc_pos: tuple[int, int] = (0, 0)
        
        self._setup_capture()
        self._setup_ui()
        
        print("Initialization complete.")
        print("===============================")
        print("         В Монтану!            ")
        print("===============================")
        print("Delay 1 sec")
        time.sleep(1)
        
        self.c = 0
    
    # Настройка видеопотока
    def _setup_capture(self):
        if self.config.camera.online:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Ошибка открытия видео")
        else:
            self.cap = cv2.VideoCapture(self.config.utils.video_path)
    
    # Настройка интерфейса и Callback мыши
    def _setup_ui(self):
        cv2.namedWindow(self.interface_name)
        cv2.setMouseCallback(self.interface_name, self._mouse_callback)
        
    # Отклик при нажатии кнопки мыши
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        
        if event == cv2.EVENT_LBUTTONDOWN:
            scale: int = 4
            self.click_point = np.array((y * scale, x * scale), dtype=np.int32)
            
            self.motion_started = True
            self.start_time_task = time.time()
            self.trajectory.clear()
            self.need_replan = True
            
            self.logger.reset_and_start()
            
            print(f"Целевая точка: {self.click_point}")

    # Основной цикл работы интерфейса
    def run(self):
        
        try:
            while True:
                start_time: float = time.time()
                
                # ================= ЭТАП 1: Захват видео =================
                ret, frame = self.cap.read()
                
                # cv2.imshow('video', frame)
                
                if not ret:
                    break
                
                self.key = cv2.waitKey(1) & 0xFF
                
                if self.key == ord("q"):
                    break
                
                self.camera_processor.load_frame(frame)
                
                # ================= ЭТАП 2: Калбировка системы =================
                if not self.camera_processor.system_calibrated_flag:
                    
                    # Если нажата клавиша "c" — выполняем калибровку по точкам
                    if self.key == ord("c"):
                        do_calibrate = True
                    else:
                        do_calibrate = False
                        
                    self.camera_processor.calibrate_system(do_calibrate=do_calibrate)
                    
                # ================= ЭТАП 3: Обработка потока видео =================
                else:
                    
                    # ================= ЭТАП 3.1: Обнаружение препятствий =================
                    get_cords = self.camera_processor.get_robot_attitude()
                    if get_cords is not None:
                        robot_position, robot_angular_position = get_cords  # [x, y], angle
                        self.robot.update_state(curr_pose=robot_position, angle=robot_angular_position)
                    
                    # ================= ЭТАП 3.2: Планирование маршрута =================
                    
                    if self.motion_started and not self.motion_started_flag:
                        self.motion_started_flag = True
                        self.logger.reset_and_start()
                        print("Движение начато")
                        
                    elif not self.motion_started and self.motion_started_flag:
                        self.motion_started_flag = False
                    
                    if self.click_point is None:
                        self.click_point = np.array(self.robot.curr_pose2D)
                    
                    if self.motion_started and len(self.path) == 0:
                        self.need_replan = True
                    
                    if float(np.linalg.norm(np.array(self.robot.curr_pose2D) - np.array(self.click_point))) <= 30:
                        self.motion_started = False
                    
                    if self.need_replan:
                        start_node: tuple[int, int] = (
                            self.robot.curr_pose2D[0] // self.config.grid.step,
                            self.robot.curr_pose2D[1] // self.config.grid.step
                        )
                        
                        goal_node: tuple[int, int] = (
                            self.click_point[0] // self.config.grid.step,
                            self.click_point[1] // self.config.grid.step
                            )
                        
                        self.path = self.grid_planner.astar(
                            self.camera_processor.pooled_mask,
                            start_node,
                            goal_node
                            )
                        self.grid_planner.print_path_info()
                        
                        self.path = [self.robot.curr_pose2D] + self.path[2:]
                        self.path = self.path[1::1] # [1::2] для уменьшения количества точек в сплайне
                        
                        if len(self.path) > 1:
                            self.path_pixels = self.grid_planner.path_to_pixels(self.path)
                            try:
                                self.spline_controller = SplineTrajectoryController(self.path_pixels, self.config.move.max_speed)
                                self.spline_points = self.spline_controller.get_full_path()
                            except:
                                continue
                        self.need_replan = False
                    
                        print(f"[INFO] START:{start_node}, END:{goal_node}")
                    
                        
                    # print(f"[INFO] Robot Pose: {self.robot.curr_pose2D}, Angular_position: {robot_angular_position}")
                    
                    # ================= ЭТАП 3.3: Расчёт сплайна =================
                    now = time.time()
                    dt = (now - self.prev_time)
                    self.prev_time = now
                    
                    # ================= ЭТАП 3.4: Движение по сплайну =================
                    if self.motion_started and self.spline_controller is not None:
                        pos, vel = self.spline_controller.update(dt)
                        self.calc_pos = tuple(pos)
                        # print(f"[INFO] Target_velocity:{vel}")
                        
                        if self.spline_controller.is_finished:
                            self.motion_started = False
                            print("[INFO] Траектория завершена.")
                            
                        
                        self.trajectory.append(self.robot.curr_pose2D)
                        current_time = time.time() - self.start_time_task
                        self.logger.write_log(current_time, self.robot.curr_pose2D, vel)
                        self.robot.navigate_velocity(velocity=vel, omega=0)
                        print(f"[INFO] Sended rotated velocity:{self.robot.velocity}")
                    else:
                        self.robot.navigate_velocity(velocity=(0, 0), omega=0)
                        
                self.display = self.camera_processor.get_display()
                
                
                # ================= ЭТАП 4: Визуализация =================
                if len(self.trajectory) > 1:
                    for i in range(1, len(self.trajectory)):
                        pt1 = tuple(self.trajectory[i-1][::-1])
                        pt2 = tuple(self.trajectory[i][::-1])
                        cv2.line(self.display, pt1, pt2, (255, 0, 255), 10)
                
                if len(self.path) > 1:
                    self.grid_planner.draw_path(
                            image=self.display,
                            path=self.path,
                            circle_color=(0, 128, 255),
                            line_color=(0, 128, 255),
                        )
                
                # Отрисовка начальнрй точки
                if self.click_point is not None:
                    cv2.circle(self.display, (self.click_point[1], self.click_point[0]), 25, (30, 255, 0), -1)
                
                # Отрисовка начальнрй точки
                if len(self.trajectory) > 0:
                    cv2.circle(self.display, tuple(self.trajectory[0][::-1]), 25, (0, 128, 255), -1)
                        
                # Отрисовка точки положения робота
                cv2.circle(self.display, tuple(self.robot.curr_pose2D[::-1]), 25, (0, 0, 255), -1)

                # Отрисовка направления
                if self.click_point is not None and self.robot.curr_pose2D is not None:
                    cv2.line(
                        self.display,
                        tuple(self.robot.curr_pose2D[::-1]),
                        tuple(self.click_point[::-1]),
                        (255, 0, 255),
                        3,
                        cv2.LINE_8
                    )
                
                if self.robot.velocity is not None:
                    rep_end = (
                            int(self.robot.curr_pose2D[1] + self.robot.velocity[1] * 1e3),
                            int(self.robot.curr_pose2D[0] + self.robot.velocity[0] * 1e3)
                        )

                    cv2.arrowedLine(
                        self.display,
                        tuple(self.robot.curr_pose2D[::-1]),
                        rep_end,
                        (176, 255, 100),
                        15
                    )
                    
                p1, p2 = self.camera_processor.trans_center
                cv2.circle(self.display, (int(p2), int(p1)), 20, (30, 255, 0), -1)
                # print(f"[INFO] Trans_center{self.camera_processor.trans_center}")
                
                
                if len(self.spline_points) > 1:
                    for i in range(len(self.spline_points) - 1):
                        cv2.line(self.display, self.spline_points[i][::-1], self.spline_points[i+1][::-1], (255, 0, 0), 10)
                
                if self.calc_pos:
                    cv2.circle(self.display, tuple(map(int, self.calc_pos)), 50, (0, 30, 255), -1)
                
                if self.camera_processor.system_calibrated_flag:
                    self.display: np.ndarray = cv2.resize(
                        self.display,
                        (self.config.map_params.resolution // 4, self.config.map_params.resolution // 4)
                    )
                
                cv2.imshow(self.interface_name, self.display)
                
                self.c %= 10
                if self.c == 0:
                    print(f"Latency = {time.time() - start_time:.4f} s ")
                self.c += 1
        
        # Обработка отключения клиента
        except (ConnectionResetError, BrokenPipeError):
            print("Client disconnected.")
        
        # Освобождение ресурсов
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            
            if self.logger is not None:
                self.logger.close()
                
            if self.config.socket_params.enable:
                self.robot.disconnect()
            print("Final")

# Точка входа
def main():
    
    # Инициализация и запуск интерфейса управления
    interface = ControlInterface()
    interface.run()

# Запуск программы
if __name__ == "__main__":
    main()