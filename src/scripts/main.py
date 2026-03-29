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
            self.current_target_index = 0
            self.click_point_changed = True
            
            print(f"Целевая точка: {click_point}")

    # Основной цикл работы интерфейса
    def run(self):
        
        try:
            while True:
                # ================= ЭТАП 1: Захват видео =================
                ret, frame = self.cap.read()
                
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
                    
                    
                    # ================= ЭТАП 3.3: Расчёт сплайна =================
                    
                    
                    # ================= ЭТАП 3.4: Движение по сплайну =================
                    self.robot.navigate_velocity(velocity=(0, 0), omega=0)



                display = self.camera_processor.get_display()
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