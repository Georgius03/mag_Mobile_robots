import yaml

import cv2
import cv2.aruco as aruco
import numpy as np

from typing import Tuple, Optional, List, Dict


with open('parameters.yaml') as config_file:
    config = yaml.safe_load(config_file)


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
