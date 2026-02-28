import heapq
import yaml

import cv2
import cv2.aruco as aruco
import numpy as np

from typing import Tuple, Optional, List, Dict


with open('parameters.yaml') as config_file:
    config = yaml.safe_load(config_file)

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

# Алгоритм Дейкстры
def dijkstra(
    pooled_mask: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Алгоритм Дейкстры для двумерной карты.

    pooled_mask:
        0 — препятствие
        1 — свободная область

    Возвращает кратчайший путь.
    """

    height: int
    width: int
    height, width = pooled_mask.shape

    if pooled_mask[start] == 0 or pooled_mask[goal] == 0:
        return []

    open_set: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_set, (0.0, start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_cost: Dict[Tuple[int, int], float] = {start: 0.0}

    directions: List[Tuple[int, int]] = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
        (-1, -1), (-1, 1),
        (1, -1), (1, 1)
    ]

    while open_set:

        current_cost: float
        current: Tuple[int, int]
        current_cost, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for dy, dx in directions:

            ny: int = current[0] + dy
            nx: int = current[1] + dx

            if not (0 <= ny < height and 0 <= nx < width):
                continue

            if pooled_mask[ny, nx] == 0:
                continue

            move_cost: float = float(np.hypot(dy, dx))
            tentative_g: float = g_cost[current] + move_cost
            neighbor: Tuple[int, int] = (ny, nx)

            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))
                came_from[neighbor] = current

    return []

# Алгоритм Greedy Best-First Search (GBFS)
def greedy_best_first(
    pooled_mask: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Жадный поиск по эвристике (Greedy Best-First Search).

    Не гарантирует оптимальность.
    Очень быстрый.
    """

    height: int
    width: int
    height, width = pooled_mask.shape

    if pooled_mask[start] == 0 or pooled_mask[goal] == 0:
        return []

    open_set: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_set, (heuristic(start, goal), start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited: set = set()

    directions: List[Tuple[int, int]] = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
        (-1, -1), (-1, 1),
        (1, -1), (1, 1)
    ]

    while open_set:

        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        if current in visited:
            continue

        visited.add(current)

        for dy, dx in directions:

            ny: int = current[0] + dy
            nx: int = current[1] + dx

            if not (0 <= ny < height and 0 <= nx < width):
                continue

            if pooled_mask[ny, nx] == 0:
                continue

            neighbor: Tuple[int, int] = (ny, nx)

            if neighbor not in visited:
                heapq.heappush(
                    open_set,
                    (heuristic(neighbor, goal), neighbor)
                )
                if neighbor not in came_from:
                    came_from[neighbor] = current

    return []

# Вывод информации о пути
def print_path_info(
    path: List[Tuple[int, int]],
    grid_step: int
) -> None:
    """
    Выводит инфографику построенного пути.

    Args:
        path: список grid-узлов [(row, col), ...]
        grid_step: размер клетки сетки [px]
    """

    if len(path) == 0:
        print("\n===== PATH INFO =====")
        print("Path: NOT FOUND")
        print("=====================\n")
        return

    # --- Старт и цель ---
    start_node: Tuple[int, int] = path[0]
    goal_node: Tuple[int, int] = path[-1]

    start_px: np.ndarray = grid_to_pixel(start_node, grid_step)
    goal_px: np.ndarray = grid_to_pixel(goal_node, grid_step)

    # --- Длина пути по сетке ---
    length_grid: float = 0.0
    for i in range(1, len(path)):
        dy: int = path[i][0] - path[i - 1][0]
        dx: int = path[i][1] - path[i - 1][1]
        length_grid += float(np.hypot(dy, dx))

    # --- Перевод в пиксели ---
    length_px: float = length_grid * grid_step

    # --- Прямое расстояние ---
    direct_dist_px: float = float(
        np.linalg.norm(goal_px.astype(np.float32) -
                       start_px.astype(np.float32))
    )

    curvature_ratio: float = (
        length_px / direct_dist_px
        if direct_dist_px > 1e-6 else 0.0
    )

    # --- Инфографика ---
    print("\n========== PATH INFO ==========")
    print(f"Nodes count        : {len(path)}")
    print(f"Start (px)         : ({start_px[0]:.1f}, {start_px[1]:.1f})")
    print(f"Goal  (px)         : ({goal_px[0]:.1f}, {goal_px[1]:.1f})")
    print("--------------------------------")
    print(f"Path length (grid) : {length_grid:.3f} cells")
    print(f"Path length (px)   : {length_px:.3f} px")
    print(f"Direct distance    : {direct_dist_px:.3f} px")
    print("--------------------------------")
    print(f"Curvature ratio    : {curvature_ratio:.3f}")
    print("================================\n")