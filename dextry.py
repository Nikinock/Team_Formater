import json
import networkx as nx
from collections import defaultdict, deque
from itertools import combinations

def find_longest_path(graph, start, end, visited_edges=None):
    """
    Находит длиннейший путь в неориентированном графе от start до end,
    не используя повторно рёбра.
    """
    if visited_edges is None:
        visited_edges = set()
    
    if start == end:
        return 0, []
    
    max_path_length = float('-inf')
    max_path = []
    
    # Перебираем всех соседей текущей вершины
    for neighbor, weight in graph[start]:
        # Создаем уникальный идентификатор ребра (сортируем вершины для неориентированного графа)
        edge = tuple(sorted([start, neighbor]))
        
        # Проверяем, не использовали ли мы уже это ребро
        if edge not in visited_edges:
            # Добавляем ребро в посещенные
            visited_edges.add(edge)
            
            # Рекурсивно ищем путь от соседа до конечной вершины
            path_length, path = find_longest_path(graph, neighbor, end, visited_edges)
            
            # Если нашли путь и он длиннее текущего максимального
            if path_length != float('-inf') and path_length + weight > max_path_length:
                max_path_length = path_length + weight
                max_path = [start] + path
            
            # Удаляем ребро из посещенных для других путей
            visited_edges.remove(edge)
    
    if max_path_length == float('-inf'):
        return float('-inf'), []
    
    return max_path_length, max_path

def find_max_path_in_graph(graph):
    """
    Находит максимальный путь между всеми парами вершин в неориентированном графе.
    Возвращает кортеж (максимальное расстояние, путь)
    """
    max_distance = float('-inf')
    max_path_info = (None, None)  # (расстояние, путь)
    
    # Перебираем все пары вершин
    for start_vertex in graph:
        for end_vertex in graph:
            if start_vertex != end_vertex:
                # Находим длиннейший путь между текущей парой вершин
                distance, path = find_longest_path(graph, start_vertex, end_vertex)
                
                if distance != float('-inf') and distance > max_distance:
                    max_distance = distance
                    max_path_info = (distance, path)
    
    return max_path_info

# Загрузка исходного графа
G = nx.read_gml("ml_ontology_graph_updated.gml")

# Создание нового графа в нужном формате (неориентированного)
graph = {}

# Преобразование графа в нужный формат с двунаправленными рёбрами
for node in G.nodes():
    # Получаем все соседние узлы и их веса
    neighbors = []
    for neighbor in G.neighbors(node):
        # Получаем вес ребра (co_weight)
        weight = G[node][neighbor].get('co_weight', 1)
        # Добавляем соседа и его вес в список
        neighbors.append((neighbor, weight))
    
    # Добавляем узел и его соседей в граф
    graph[node] = neighbors

# Сохранение графа в файл
with open('graph_structure.py', 'w', encoding='utf-8') as f:
    f.write("graph = {\n")
    for node, neighbors in graph.items():
        f.write(f"    '{node}': {neighbors},\n")
    f.write("}\n")

print("\nГраф сохранен в файл 'graph_structure.py'")

# Пример использования алгоритма поиска длиннейшего пути
if __name__ == "__main__":
    print("\nПоиск максимального пути в неориентированном графе...")
    
    # Находим максимальный путь
    max_distance, max_path = find_max_path_in_graph(graph)
    
    if max_distance != float('-inf'):
        print(f"\nНайден максимальный путь:")
        print(f"От вершины: '{max_path[0]}'")
        print(f"До вершины: '{max_path[-1]}'")
        print(f"Длина пути: {max_distance:.4f}")
        print("\nПолный путь:")
        print(" -> ".join(max_path))
    else:
        print("\nВ графе нет достижимых путей")

