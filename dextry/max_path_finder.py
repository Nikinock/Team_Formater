import json
import heapq
from itertools import combinations
from collections import defaultdict

def max_dijkstra(graph, start_node, max_path_length=None):
    """
    Модифицированный алгоритм Дейкстры для поиска путей с максимальным весом.
    Предотвращает циклы в путях и позволяет ограничить длину пути.
    
    Args:
        graph: словарь, представляющий граф
        start_node: начальная вершина
        max_path_length: максимальная длина пути (None для неограниченной длины)
    """
    # Инициализация с отрицательной бесконечностью
    distances = {vertex: float('-infinity') for vertex in graph}
    distances[start_node] = 0
    priority_queue = [(0, start_node, [start_node])]  # Добавляем текущий путь в очередь
    previous_nodes = {vertex: None for vertex in graph}
    visited_paths = set()  # Множество для отслеживания уже посещенных путей

    while priority_queue:
        current_distance, current_vertex, current_path = heapq.heappop(priority_queue)
        current_distance = -current_distance  # Преобразуем обратно в положительное значение
        
        # Проверяем длину пути
        if max_path_length and len(current_path) > max_path_length:
            continue
            
        # Если текущий путь меньше максимального, пропускаем
        if current_distance < distances[current_vertex]:
            continue
            
        for neighbor, weight in graph[current_vertex].items():
            # Проверяем, не создаст ли добавление соседа цикл
            if neighbor in current_path:
                continue
                
            # Создаем новый путь
            new_path = current_path + [neighbor]
            # Создаем уникальный ключ для пути
            path_key = tuple(new_path)
            
            # Проверяем, не посещали ли мы уже этот путь
            if path_key in visited_paths:
                continue
                
            visited_paths.add(path_key)
            
            # Считаем новый вес пути
            distance = current_distance + weight
            if distance > distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_vertex
                # Добавляем в очередь с новым путем
                heapq.heappush(priority_queue, (-distance, neighbor, new_path))
                
    return distances, previous_nodes

def find_max_path(graph_file, max_path_length=None):
    """
    Находит максимальный путь между всеми парами вершин в графе.
    Предотвращает циклы и позволяет ограничить длину пути.
    
    Args:
        graph_file: путь к JSON файлу с графом
        max_path_length: максимальная длина пути (None для неограниченной длины)
    
    Returns:
        tuple: (max_distance, max_path, start_node, end_node)
    """
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    
    max_distance = float('-infinity')
    max_path = []
    max_start = None
    max_end = None
    
    nodes = list(graph.keys())
    total_pairs = len(list(combinations(nodes, 2)))
    print(f"Анализ {total_pairs} пар вершин...")
    if max_path_length:
        print(f"Максимальная длина пути: {max_path_length}")
    
    for i, (start, end) in enumerate(combinations(nodes, 2), 1):
        if i % 100 == 0:  # Прогресс каждые 100 пар
            print(f"\nОбработано {i}/{total_pairs} пар...")
            
        distances, previous_nodes = max_dijkstra(graph, start, max_path_length)
        
        if distances[end] > max_distance:
            max_distance = distances[end]
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = previous_nodes[current]
            path.reverse()
            max_path = path
            max_start = start
            max_end = end
            print(f"\nНайден новый максимальный путь!")
            print(f"От '{start}' до '{end}'")
            print(f"Вес пути: {max_distance:.4f}")
            print(f"Длина пути: {len(path)}")
            print(f"Путь: {' -> '.join(path)}")
    
    return max_distance, max_path, max_start, max_end

def print_max_path(distance, path, start, end):
    """Выводит информацию о найденном максимальном пути"""
    print("\nМаксимальный путь в графе:")
    print(f"От '{start}' до '{end}'")
    print(f"Вес пути: {distance:.4f}")
    print(f"Путь: {' -> '.join(path)}")
    print(f"Количество вершин в пути: {len(path)}")

if __name__ == "__main__":
    try:
        # Можно указать максимальную длину пути, например:
        # max_distance, max_path, start, end = find_max_path("graph_weights.json", max_path_length=5)
        max_distance, max_path, start, end = find_max_path("graph_weights.json")
        print_max_path(max_distance, max_path, start, end)
    except FileNotFoundError:
        print("Ошибка: Файл с графом не найден")
    except json.JSONDecodeError:
        print("Ошибка: Неверный формат JSON файла")
    except Exception as e:
        print(f"Произошла ошибка: {e}") 