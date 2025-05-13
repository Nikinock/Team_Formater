import networkx as nx
import matplotlib.pyplot as plt

# Загрузка графа из GML-файла
G = nx.read_gml("ml_ontology_graph_updated.gml")

# Настройка размера графика
plt.figure(figsize=(15, 15))

# Выбор layout для красивого расположения узлов
pos = nx.spring_layout(G, k=0.15, iterations=20)

# Визуализация графа
nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, edge_color='gray')

plt.title("Визуализация графа из GML")
plt.show()