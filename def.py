import networkx as nx
from heapq import heappush, heappop
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

def find_max_weight_path(G: nx.Graph, start: str, end: str) -> Tuple[float, List[str]]:
    """
    Находит максимальный взвешенный путь между двумя вершинами в графе по co_weight.
    Возвращает (вес, путь). Если путь не найден, возвращает (0, []).
    """
    if not G.has_node(start) or not G.has_node(end):
        return 0, []
    distances = {node: float('-inf') for node in G.nodes()}
    distances[start] = 0
    previous = {node: None for node in G.nodes()}
    queue = [(0, start)]
    while queue:
        current_dist, current = heappop(queue)
        current_dist = -current_dist
        if current == end:
            break
        if current_dist < distances[current]:
            continue
        for neighbor in G.neighbors(current):
            if G.has_edge(current, neighbor):
                weight = G[current][neighbor].get('co_weight', 1)
                new_dist = current_dist + weight
                if new_dist > distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heappush(queue, (-new_dist, neighbor))
    if distances[end] == float('-inf'):
        return 0, []
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    return distances[end], path[::-1]

def print_path_info(G: nx.Graph, start: str, end: str) -> None:
    weight, path = find_max_weight_path(G, start, end)
    if not path:
        print(f"Путь между '{start}' и '{end}' не найден")
        return
    print(f"\nМаксимальный путь между '{start}' и '{end}':")
    print(f"Общий вес пути: {weight:.4f}")
    print("Последовательность вершин:")
    for i, node in enumerate(path):
        if i < len(path) - 1:
            edge_weight = G[node][path[i+1]].get('co_weight', 1)
            print(f"{node} --[{edge_weight:.4f}]--> ", end='')
        else:
            print(node)

def print_all_nodes(G: nx.Graph):
    print("\nВсе вершины графа:")
    for node in G.nodes():
        print(node)

def analyze_graph_structure(G: nx.Graph) -> None:
    """
    Анализирует структуру графа и выводит статистику.
    """
    print("\nАнализ структуры графа:")
    print(f"Количество вершин: {len(G.nodes())}")
    print(f"Количество рёбер: {len(G.edges())}")
    
    # Анализ отношений
    relations = defaultdict(int)
    for u, v, data in G.edges(data=True):
        rel = data.get('relation', 'UNKNOWN')
        relations[rel] += 1
    
    print("\nРаспределение отношений:")
    for rel, count in relations.items():
        print(f"{rel}: {count}")
    
    # Анализ весов
    weights = [data.get('co_weight', 1) for _, _, data in G.edges(data=True)]
    if weights:
        print(f"\nСтатистика весов:")
        print(f"Минимальный вес: {min(weights):.4f}")
        print(f"Максимальный вес: {max(weights):.4f}")
        print(f"Средний вес: {sum(weights)/len(weights):.4f}")

# Пример использования:
if __name__ == "__main__":
    # Загрузка графа
    G = nx.read_gml("ml_ontology_graph_updated.gml")
    
    # Анализ структуры графа
    analyze_graph_structure(G)
    
    print(f"В графе {len(G.nodes())} вершин.")
    print_all_nodes(G)
    
    # Пример: замените на нужные вам вершины
    start = input("\nВведите начальную вершину: ").strip()
    end = input("Введите конечную вершину: ").strip()
    print_path_info(G, start, end)
