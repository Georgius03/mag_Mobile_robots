import heapq

import cv2
import cv2.aruco as aruco
import numpy as np

from typing import Tuple, Optional, List, Dict

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
