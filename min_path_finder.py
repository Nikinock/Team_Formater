import json
import networkx as nx
from collections import defaultdict, deque
from typing import Tuple, List, Optional

def find_min_path(graph: dict, start: str, end: str, visited_edges: Optional[set] = None) -> Tuple[float, List[str]]:
    """
    Находит минимальный путь в неориентированном графе от start до end,
    не используя повторно рёбра.
    
    Args:
        graph: словарь {u: [(v, weight), ...]}
        start: начальная вершина
        end: конечная вершина
        visited_edges: множество уже использованных рёбер
    
    Returns:
        Tuple[float, List[str]]: (длина пути, список вершин в пути)
        Если путь не найден, возвращает (float('inf'), [])
    """
    if visited_edges is None:
        visited_edges = set()
    
    if start == end:
        return 0, []
    
    min_path_length = float('inf')
    min_path = []
    
    # Перебираем всех соседей текущей вершины
    for neighbor, weight in graph[start]:
        # Создаем уникальный идентификатор ребра (сортируем вершины для неориентированного графа)
        edge = tuple(sorted([start, neighbor]))
        
        # Проверяем, не использовали ли мы уже это ребро
        if edge not in visited_edges:
            # Добавляем ребро в посещенные
            visited_edges.add(edge)
            
            # Рекурсивно ищем путь от соседа до конечной вершины
            path_length, path = find_min_path(graph, neighbor, end, visited_edges)
            
            # Если нашли путь и он короче текущего минимального
            if path_length != float('inf') and path_length + weight < min_path_length:
                min_path_length = path_length + weight
                min_path = [start] + path
            
            # Удаляем ребро из посещенных для других путей
            visited_edges.remove(edge)
    
    if min_path_length == float('inf'):
        return float('inf'), []
    
    return min_path_length, min_path

def find_path_between_vertices(graph: dict, start: str, end: str) -> Tuple[float, List[str]]:
    """
    Находит минимальный путь между двумя заданными вершинами в графе.
    
    Args:
        graph: словарь {u: [(v, weight), ...]}
        start: начальная вершина
        end: конечная вершина
    
    Returns:
        Tuple[float, List[str]]: (длина пути, список вершин в пути)
        Если путь не найден, возвращает (float('inf'), [])
    """
    if start not in graph or end not in graph:
        return float('inf'), []
    
    return find_min_path(graph, start, end)

def load_graph_from_file(filename: str) -> dict:
    """
    Загружает граф из файла graph_structure_new.py
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            local_vars = {}
            exec(f.read(), {}, local_vars)
            return local_vars.get('graph', {})
    except Exception as e:
        print(f"Ошибка при загрузке графа: {e}")
        return {}

if __name__ == "__main__":
    # Загружаем граф из файла
    graph = load_graph_from_file('graph_structure_new.py')
    
    if not graph:
        print("Не удалось загрузить граф")
        exit(1)
    
    # Запрашиваем у пользователя начальную и конечную вершины
    print("\nДоступные вершины:")
    vertices = sorted(graph.keys())
    for i, vertex in enumerate(vertices, 1):
        print(f"{i}. {vertex}")
    
    try:
        start_idx = int(input("\nВведите номер начальной вершины: ")) - 1
        end_idx = int(input("Введите номер конечной вершины: ")) - 1
        
        if 0 <= start_idx < len(vertices) and 0 <= end_idx < len(vertices):
            start_vertex = vertices[start_idx]
            end_vertex = vertices[end_idx]
            
            # Ищем минимальный путь
            path_length, path = find_path_between_vertices(graph, start_vertex, end_vertex)
            
            if path_length != float('inf'):
                print(f"\nНайден минимальный путь:")
                print(f"От вершины: '{start_vertex}'")
                print(f"До вершины: '{end_vertex}'")
                print(f"Длина пути: {path_length:.4f}")
                print("\nПолный путь:")
                print(" -> ".join(path))
            else:
                print(f"\nПуть между вершинами '{start_vertex}' и '{end_vertex}' не найден")
        else:
            print("Неверный номер вершины")
    except ValueError:
        print("Пожалуйста, введите корректный номер вершины")
    except Exception as e:
        print(f"Произошла ошибка: {e}") 