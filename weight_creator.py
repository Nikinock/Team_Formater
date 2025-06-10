import json
import networkx as nx
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Пороговые значения для метрик
JACCARD_THRESHOLD = 0.002
LIFT_THRESHOLD = 1
CHI_SQUARE_THRESHOLD = 3.841


def png_graph_creator():
    plt.figure(figsize=(15, 6))
    # График распределения меры Жаккара
    plt.subplot(1, 2, 1)
    plt.hist(jaccard_weights, bins=50, alpha=0.7, color='blue', range=(0, 0.02))
    plt.title('Распределение меры Жаккара')
    plt.xlabel('Значение меры Жаккара')
    plt.ylabel('Количество пар')
    plt.xlim(0, 0.02)
    plt.grid(True, alpha=0.3)
    
    # График распределения метрики Lift
    plt.subplot(1, 2, 2)
    plt.hist(lift_weights, bins=50, alpha=0.7, color='green', range=(0, 20))
    plt.title('Распределение метрики Lift')
    plt.xlabel('Значение метрики Lift')
    plt.ylabel('Количество пар')
    plt.xlim(0, 3)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_distribution.png')
    plt.close()


# Загрузка графа
G = nx.read_gml("ml_ontology_graph_normalized.gml")
graph_terms = list(G.nodes)

# Загрузка упоминаний
with open("entity_counts_by_parts.json", "r", encoding="utf-8") as f:
    entity_data = json.load(f)


# Подсчёт упоминаний
individual_counts = defaultdict(int)
co_occurrence_counts = defaultdict(int)

for text in entity_data:
    for part in text["parts"]:
        for term in part["terms"].keys():
            if term in graph_terms:
                individual_counts[term] += part["terms"][term]
                
        for t1, t2 in combinations(part["terms"].keys(), 2):
            key = tuple(sorted((t1, t2)))
            co_occurrence_counts[key] += 1

# Добавление рёбер в граф + сбор статистики по весам
new_edges_added = 0
updated_edges = 0
weights = defaultdict(dict)  # Используем defaultdict для автоматической инициализации

# Подсчёт весов
for (t1, t2), co_count in co_occurrence_counts.items():
    min_count = min(individual_counts[t1], individual_counts[t2])
    weight = co_count / min_count if min_count > 0 else 0
    weights[t1][t2] = weight

# Статистика по весам
if weights:
    all_weights = [w for t1 in weights.values() for w in t1.values()]
    max_weight = max(all_weights)
    min_weight = min(all_weights)
    avg_weight = sum(all_weights) / len(all_weights)
else:
    max_weight = min_weight = avg_weight = 0

# Расчет меры Жаккара, Lift и хи-квадрат
jaccard_weights = defaultdict(dict)
lift_weights = defaultdict(dict)
chi_square_weights = defaultdict(dict)
total_documents = len(entity_data)

for (t1, t2), co_count in co_occurrence_counts.items():
    n1 = individual_counts[t1]
    n2 = individual_counts[t2]
    
    # Расчет меры Жаккара
    jaccard = co_count / (n1 + n2 - co_count) if (n1 + n2 - co_count) > 0 else 0
    jaccard_weights[t1][t2] = jaccard
    
    # Расчет метрики Lift
    expected_co_occurrence = (n1 * n2) / total_documents
    lift = (co_count * total_documents) / (n1 * n2) if (n1 * n2) > 0 else 0
    lift_weights[t1][t2] = lift
    
    # Расчет хи-квадрат
    # Создаем таблицу сопряженности 2x2
    # [a b]
    # [c d]
    a = co_count  # O_ij
    b = n1 - co_count
    c = n2 - co_count
    d = total_documents - n1 - n2 + co_count
    
    # Ожидаемые частоты
    E_a = (n1 * n2) / total_documents
    E_b = (n1 * (total_documents - n2)) / total_documents
    E_c = ((total_documents - n1) * n2) / total_documents
    E_d = ((total_documents - n1) * (total_documents - n2)) / total_documents
    
    # Расчет хи-квадрат
    chi_square = ((a - E_a)**2 / E_a + 
                  (b - E_b)**2 / E_b + 
                  (c - E_c)**2 / E_c + 
                  (d - E_d)**2 / E_d) if (E_a > 0 and E_b > 0 and E_c > 0 and E_d > 0) else 0
    
    chi_square_weights[t1][t2] = chi_square

# Изменяем цикл для работы со словарями
for t1 in weights:
    for t2 in weights[t1]:
        weight = weights[t1][t2]
        if (jaccard_weights[t1][t2] >= JACCARD_THRESHOLD and 
            lift_weights[t1][t2] >= LIFT_THRESHOLD and 
            chi_square_weights[t1][t2] >= CHI_SQUARE_THRESHOLD):
            if G.has_edge(t1, t2):
                G[t1][t2]["co_weight"] = 1 - weight
                updated_edges += 1
            else:
                G.add_edge(t1, t2, co_weight=1 - weight)
                new_edges_added += 1

for u, v, g in G.edges(data=True):
    if 'co_weight' not in g or not g['co_weight']:
        g['co_weight'] = 1

png_graph_creator()

# Вывод статистики по обеим метрикам
if jaccard_weights:
    all_jaccard = [w for t1 in jaccard_weights.values() for w in t1.values()]
    max_jaccard = max(all_jaccard)
    min_jaccard = min(all_jaccard)
    avg_jaccard = sum(all_jaccard) / len(all_jaccard)
else:
    max_jaccard = min_jaccard = avg_jaccard = 0

if lift_weights:
    all_lift = [w for t1 in lift_weights.values() for w in t1.values()]
    max_lift = max(all_lift)
    min_lift = min(all_lift)
    avg_lift = sum(all_lift) / len(all_lift)
else:
    max_lift = min_lift = avg_lift = 0

print("\n--- Статистика по мере Жаккара ---")
print(f"Максимальное значение: {max_jaccard:.4f}")
print(f"Минимальное ненулевое значение: {min_jaccard:.4f}")
print(f"Среднее значение: {avg_jaccard:.4f}")

print("\n--- Статистика по метрике Lift ---")
print(f"Максимальное значение: {max_lift:.4f}")
print(f"Минимальное ненулевое значение: {min_lift:.4f}")
print(f"Среднее значение: {avg_lift:.4f}")

# Вывод статистики по хи-квадрат
if chi_square_weights:
    all_chi = [w for t1 in chi_square_weights.values() for w in t1.values()]
    max_chi = max(all_chi)
    min_chi = min(all_chi)
    avg_chi = sum(all_chi) / len(all_chi)
    significant_chi_count = sum(1 for x in all_chi if x > CHI_SQUARE_THRESHOLD)
    p_values = [1 - chi2.cdf(x, 1) for x in all_chi if x > 0]
else:
    max_chi = min_chi = avg_chi = 0
    significant_chi_count = 0
    p_values = []

print("\n--- Статистика по критерию хи-квадрат ---")
print(f"Максимальное значение: {max_chi:.4f}")
print(f"Минимальное ненулевое значение: {min_chi:.4f}")
print(f"Среднее значение: {avg_chi:.4f}")
print(f"Количество значимых связей (χ² > {CHI_SQUARE_THRESHOLD}): {significant_chi_count}")
if p_values:
    print(f"Среднее p-value для значимых связей: {sum(p_values)/len(p_values):.4f}")

# Анализ наиболее значимых связей по хи-квадрат
significant_chi_pairs = []
for t1 in chi_square_weights:
    for t2 in chi_square_weights[t1]:
        chi = chi_square_weights[t1][t2]
        if chi > CHI_SQUARE_THRESHOLD:
            significant_chi_pairs.append((t1, t2, chi))

significant_chi_pairs.sort(key=lambda x: x[2], reverse=True)

print("\n--- Топ-5 наиболее значимых пар по критерию хи-квадрат ---")
for t1, t2, chi in significant_chi_pairs[:5]:
    p_value = 1 - chi2.cdf(chi, 1)
    print(f"{t1} - {t2}:")
    print(f"  χ² = {chi:.4f}")
    print(f"  p-value = {p_value:.4f}")

# Сохраняем граф с новыми весами
nx.write_gml(G, "ml_ontology_graph_updated.gml")

# Вывод статистики
print("Обновление завершено!")
print(f"Сущностей в графе: {len(G.nodes)}")
print(f"Новых рёбер добавлено: {new_edges_added}")
print(f"Существующих рёбер обновлено: {updated_edges}")
print(f"Совместных упоминаний найдено: {len(co_occurrence_counts)}")
print("--- Статистика по весам ---")
print(f"Максимальный вес: {max_weight:.4f}")
print(f"Минимальный ненулевой вес: {min_weight:.4f}")
print(f"Средний вес: {avg_weight:.4f}")

def find_max_paths(G, max_paths=5):
    """Поиск максимальных путей в графе с учетом весов"""
    print("\n--- Поиск максимальных путей в графе ---")
    
    # Создаем неориентированный граф для поиска путей
    undirected_G = G.to_undirected()
    
    # Получаем все пары вершин
    all_pairs = list(combinations(undirected_G.nodes(), 2))
    max_paths_list = []
    
    print(f"Анализ {len(all_pairs)} возможных пар вершин...")
    
    for source, target in all_pairs:
        try:
            # Находим все простые пути между вершинами
            paths = list(nx.all_simple_paths(undirected_G, source, target, cutoff=3))
            
            for path in paths:
                # Считаем сумму весов для пути
                path_weight = 0
                for i in range(len(path)-1):
                    # Берем вес ребра (co_weight)
                    edge_weight = undirected_G[path[i]][path[i+1]].get('co_weight', 1)
                    path_weight += edge_weight
                
                # Сохраняем путь и его вес
                max_paths_list.append((path, path_weight))
                
        except nx.NetworkXNoPath:
            continue
    
    # Сортируем пути по весу (по убыванию)
    max_paths_list.sort(key=lambda x: x[1], reverse=True)
    
    # Выводим топ-N путей
    print(f"\nТоп-{max_paths} максимальных путей:")
    for i, (path, weight) in enumerate(max_paths_list[:max_paths], 1):
        path_str = " -> ".join(path)
        print(f"{i}. Путь: {path_str}")
        print(f"   Общий вес: {weight:.4f}")
        print(f"   Средний вес на ребро: {weight/(len(path)-1):.4f}")
        print()

if __name__ == "__main__":
    # ... existing code ...
    
    # После сохранения графа ищем максимальные пути
    find_max_paths(G)
