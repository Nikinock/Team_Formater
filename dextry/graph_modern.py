import networkx as nx
from itertools import combinations
from collections import defaultdict
import json
from dextry import dijkstra

G = nx.read_gml("graph.gml")

nodes = G.nodes()
graph = defaultdict(dict)

for t0, t1 in combinations(nodes, 2):
    if G.has_edge(t0, t1):
        weight = G[t0][t1]["co_weight"]
        graph[t0][t1] = weight
        graph[t1][t0] = weight
    if G.has_edge(t1, t0):
        weight = G[t1][t0]["co_weight"]
        graph[t1][t0] = weight
        graph[t0][t1] = weight
print("Граф:")
for node, edges in graph.items():
    print(f"{node}: {edges}")

# Сохраняем граф в JSON файл
with open("graph_weights.json", "w", encoding="utf-8") as f:
    json.dump(dict(graph), f, ensure_ascii=False, indent=2)

print("\nГраф сохранен в файл 'graph_weights.json'")

# Тестируем алгоритм Дейкстры
print("\nТестирование алгоритма Дейкстры:")
start_node = list(graph.keys())[0]  # Берем первую вершину как стартовую
distances, previous_nodes = dijkstra(graph, start_node)

print(f"\nКратчайшие пути от вершины '{start_node}':")
for node, distance in distances.items():
    if distance != float('infinity'):
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = previous_nodes[current]
        path.reverse()
        print(f"До '{node}': расстояние = {distance:.4f}, путь = {' -> '.join(path)}")
    else:
        print(f"До '{node}': недостижим")
