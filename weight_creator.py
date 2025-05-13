import json
import networkx as nx
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import sys

# Получение порогового значения n из аргументов командной строки
if len(sys.argv) != 2:
    print("Использование: python weight_creator.py <пороговое_значение_n>")
    sys.exit(1)

try:
    n = float(sys.argv[1])
except ValueError:
    print("Пороговое значение должно быть числом")
    sys.exit(1)

def normalize(term):
    return term.strip().lower()

# Загрузка графа
G = nx.read_gml("ml_ontology_graph_normalized.gml")

# Создание словаря нормализованных названий
graph_terms = list(G.nodes)
normalized_to_original = {normalize(t): t for t in graph_terms}

# Загрузка упоминаний
with open("entity_counts_by_parts.json", "r", encoding="utf-8") as f:
    entity_data = json.load(f)

# Подсчёт упоминаний
individual_counts = defaultdict(int)
co_occurrence_counts = defaultdict(int)

for text in entity_data:
    for part in text["parts"]:
        # нормализуем термины из части
        normalized_terms = []
        for term in part["terms"].keys():
            norm = normalize(term)
            if norm in normalized_to_original:
                normalized_terms.append(norm)
                individual_counts[norm] += part["terms"][term]

        for t1, t2 in combinations(normalized_terms, 2):
            key = tuple(sorted((t1, t2)))
            co_occurrence_counts[key] += 1

# Добавление рёбер в граф + сбор статистики по весам
new_edges_added = 0
updated_edges = 0
weights = []

for (t1_norm, t2_norm), co_count in co_occurrence_counts.items():
    t1 = normalized_to_original[t1_norm]
    t2 = normalized_to_original[t2_norm]

    min_count = min(individual_counts[t1_norm], individual_counts[t2_norm])
    weight = co_count / min_count if min_count > 0 else 0
    final_weight = 1 - weight  # Новый итоговый вес

    if weight > n:
        weights.append(final_weight)

    if G.has_edge(t1, t2):
        if weight > n:
            G[t1][t2]["co_weight"] = final_weight
            updated_edges += 1
    else:
        if weight > n:
            G.add_edge(t1, t2, co_weight=final_weight)
            new_edges_added += 1

# Всем не обновлённым рёбрам присваиваем вес 1
for u, v, data in G.edges(data=True):
    if "co_weight" not in data:
        data["co_weight"] = 1

# Статистика по весам
if weights:
    max_weight = max(weights)
    min_weight = min(weights)
    avg_weight = sum(weights) / len(weights)
else:
    max_weight = min_weight = avg_weight = 0

# Расчет меры Жаккара, Lift и хи-квадрат
jaccard_weights = []
lift_weights = []
chi_square_weights = []
total_documents = len(entity_data)  # Общее количество документов
critical_value = 3.841  # Критическое значение для α=0.05 и df=1

for (t1_norm, t2_norm), co_count in co_occurrence_counts.items():
    t1 = normalized_to_original[t1_norm]
    t2 = normalized_to_original[t2_norm]
    
    n1 = individual_counts[t1_norm]
    n2 = individual_counts[t2_norm]
    
    # Расчет меры Жаккара
    jaccard = co_count / (n1 + n2 - co_count) if (n1 + n2 - co_count) > 0 else 0
    jaccard_weights.append(jaccard)
    
    # Расчет метрики Lift
    expected_co_occurrence = (n1 * n2) / total_documents
    lift = (co_count * total_documents) / (n1 * n2) if (n1 * n2) > 0 else 0
    lift_weights.append(lift)
    
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
    
    chi_square_weights.append(chi_square)
    
    # Проверяем все условия перед добавлением в граф
    if jaccard >= 0.02 and lift >= 20 and chi_square >= 3:
        if G.has_edge(t1, t2):
            G[t1][t2]["jaccard_weight"] = jaccard
            G[t1][t2]["lift_weight"] = lift
            G[t1][t2]["chi_square"] = chi_square
            G[t1][t2]["co_weight"] = weight
            updated_edges += 1
        else:
            G.add_edge(t1, t2, jaccard_weight=jaccard, lift_weight=lift, 
                      chi_square=chi_square, co_weight=weight)
            new_edges_added += 1

# Создание графиков
plt.figure(figsize=(15, 6))

# График распределения меры Жаккара
plt.subplot(1, 2, 1)
plt.hist(jaccard_weights, bins=50, alpha=0.7, color='blue')
plt.title('Распределение меры Жаккара')
plt.xlabel('Значение меры Жаккара')
plt.ylabel('Количество пар')
plt.grid(True, alpha=0.3)

# График распределения метрики Lift
plt.subplot(1, 2, 2)
plt.hist(lift_weights, bins=50, alpha=0.7, color='green')
plt.title('Распределение метрики Lift')
plt.xlabel('Значение метрики Lift')
plt.ylabel('Количество пар')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('metrics_distribution.png')
plt.close()

# Вывод статистики по обеим метрикам
if jaccard_weights:
    max_jaccard = max(jaccard_weights)
    min_jaccard = min(jaccard_weights)
    avg_jaccard = sum(jaccard_weights) / len(jaccard_weights)
else:
    max_jaccard = min_jaccard = avg_jaccard = 0

if lift_weights:
    max_lift = max(lift_weights)
    min_lift = min(lift_weights)
    avg_lift = sum(lift_weights) / len(lift_weights)
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
    max_chi = max(chi_square_weights)
    min_chi = min(chi_square_weights)
    avg_chi = sum(chi_square_weights) / len(chi_square_weights)
    significant_chi_count = sum(1 for x in chi_square_weights if x > critical_value)
    p_values = [1 - chi2.cdf(x, 1) for x in chi_square_weights if x > 0]
else:
    max_chi = min_chi = avg_chi = 0
    significant_chi_count = 0
    p_values = []

print("\n--- Статистика по критерию хи-квадрат ---")
print(f"Максимальное значение: {max_chi:.4f}")
print(f"Минимальное ненулевое значение: {min_chi:.4f}")
print(f"Среднее значение: {avg_chi:.4f}")
print(f"Количество значимых связей (χ² > {critical_value}): {significant_chi_count}")
if p_values:
    print(f"Среднее p-value для значимых связей: {sum(p_values)/len(p_values):.4f}")

# Анализ наиболее значимых связей по хи-квадрат
significant_chi_pairs = [(t1, t2, chi) for (t1, t2), chi in zip(co_occurrence_counts.keys(), chi_square_weights) if chi > critical_value]
significant_chi_pairs.sort(key=lambda x: x[2], reverse=True)

print("\n--- Топ-5 наиболее значимых пар по критерию хи-квадрат ---")
for t1, t2, chi in significant_chi_pairs[:5]:
    p_value = 1 - chi2.cdf(chi, 1)
    print(f"{normalized_to_original[t1]} - {normalized_to_original[t2]}:")
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
