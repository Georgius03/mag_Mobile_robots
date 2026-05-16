from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from math import radians

# Параметры подключения к роботу
class SocketParams(BaseModel):
    IP_ADDRESS: str = "192.168.0.1"                                         # IP SocketServer
    PORT: int = 80                                                          # Порт SocketServer
    enable: (bool | int) = 1                                                # Отк./вкл. для тестирования (0/1) чтобы не производилось подключение к роботу

# Параметры камеры
class CameraParams(BaseModel):
    online: (bool | int) = 1                                                # Переключение на подгрузку кадров из видео-записьи/трансляцитю с камеры (0/1)
    H: float = 250.0                                                        # Высота от поля до камеры
    h: float = 24.3                                                         # Высота препятствия
    sharpening: (bool | int) = 0                                            # Повышение резкости (вкл. при проблемах с фокусом на камере)
    equalize_hist: (bool | int) = 0                                         # Выравнивание гистограммы
    cam_shape_width: int = 640                                              # Разрешение камеры (по умолчанию 640)
    cam_shape_height: int = 480                                             # Разрешение камеры (по умолчанию 480)

# Параметры полигона
class MapParams(BaseModel):
    resolution: int = 2200                                                  # Размер поля в мм (по краям калибровочных ArUco маркеров)
    left_up: tuple[int, int] = (24, 147)                                    # Стартовые положения (лево верх)
    right_up: tuple[int, int] = (24, 588)                                   # Стартовые положения (право верх)
    right_down: tuple[int, int] = (438, 564)                                # Стартовые положения (право низ)
    left_down: tuple[int, int] = (440, 166)                                 # Стартовые положения (лево низ)

# Параметры движения работа
class MoveParams(BaseModel):
    dist_stop: int = 50                                                     # Расстояние для остановки при движении к точке (мм)
    max_speed: float = 0.08                                                 # Максимальная скорость Robotino (м/с)
    k_prop: float = 1.0                                                     # Пропорциональный коэффициент
    filter_gain: float = 0.2                                                # Коэффициент фильтрации скорости
    ArUco_angle: float = radians(180)                                       # Угол поворота для коррекции ориентации робота (для ArUco маркера)
    spline_smoothing_lambda: float = 1e-3                                   # Параметр сглаживания для сплайна (чем меньше, тем более гладкий сплайн, но может отклоняться от точек)

# Параметры сетки и препятствий
class GridParams(BaseModel):
    step: int = 44                                                          # Шаг дискретизации пикселей n*px/px (22, 44, 55, 110, 220, 440) желательно выбирать кратно размеру поля
    robot_radius: int = 340                                                 # Радиус робота (+ допуск из-за искажений камеры при проецировании)
    WALL_WIDTH: int = 200                                                   # Толщина стенки по контуру рабочего поля (для 1 задачи)

# Пути к директориям
class UtilsParams(BaseModel):
    log_dir: str = "logs/"                                                  # Путь к папке с логами
    video_path: str = "video/vid.mkv"                                       # Путь к видео при отключении online в camera

# Оркестратор параметров
class Settings(BaseSettings):
    socket_params: SocketParams = SocketParams()
    camera: CameraParams = CameraParams()
    map_params: MapParams = MapParams()
    move: MoveParams = MoveParams()
    grid: GridParams = GridParams()
    utils: UtilsParams = UtilsParams()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()