import networkx as nx
import re

def normalize_term(term):
    term = term.lower()
    term = re.sub(r'_', ' ', term)
    term = re.sub(r'(?<=\s)-|-(?=\s)|(?<=\n)-|-(?=\n)', '', term)
    return term.strip()

def update_graph_terms(input_gml, output_gml):
    # Загрузка графа
    G = nx.read_gml(input_gml)

    # Словарь соответствия: старое -> новое имя
    rename_map = {}
    for node in list(G.nodes):
        normalized = normalize_term(node)
        if normalized != node:
            # Обеспечим уникальность: если уже существует такой узел — объединяем
            if normalized in G:
                # Переносим все рёбра от старого узла к существующему
                for neighbor in list(G.neighbors(node)):
                    data = G.get_edge_data(node, neighbor)
                    if not G.has_edge(normalized, neighbor):
                        G.add_edge(normalized, neighbor, **data)
                G.remove_node(node)
            else:
                rename_map[node] = normalized

    # Переименование оставшихся узлов
    nx.relabel_nodes(G, rename_map, copy=False)

    # Сохраняем обновлённый граф
    nx.write_gml(G, output_gml)
    print(f"Граф сохранён в файл: {output_gml}")
    print(f"Переименовано узлов: {len(rename_map)}")
    print(f"Общее количество узлов: {len(G.nodes)}")

if __name__ == "__main__":
    input_gml = "ml_ontology_graph.gml"
    output_gml = "ml_ontology_graph_normalized.gml"
    update_graph_terms(input_gml, output_gml)
