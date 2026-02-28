import cv2
import cv2.aruco as aruco
import numpy as np
import yaml

from typing import Tuple, Optional, List, Dict


with open('parameters.yaml') as config_file:
    config = yaml.safe_load(config_file)

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
