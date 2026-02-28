import yaml

import cv2
import cv2.aruco as aruco
import numpy as np

from typing import Tuple, Optional, List, Dict


with open('parameters.yaml') as config_file:
    config = yaml.safe_load(config_file)

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
