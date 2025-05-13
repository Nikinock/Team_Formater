from scipy.optimize import linear_sum_assignment
import numpy as np

def match_candidates_to_project(candidates, project_requirements, Q_min, Q_max):
    """
    Распределяет кандидатов по требованиям проекта с учетом их навыков и количества мест в команде.
    Оптимизирует назначение, проверяя возможность замены кандидатов для улучшения общей оценки команды.
    Учитывает избыточное/недостаточное количество исполнителей.
    
    :param candidates: словарь {"имя": {"навык": уровень (1-5), ...}, ...}
    :param project_requirements: словарь {"навык": {"уровень": требуемый уровень (1-5), "количество": число членов команды}, ...}
    :param Q_min: минимальное количество исполнителей
    :param Q_max: максимальное количество исполнителей
    :return: список назначенных кандидатов
    """
    candidate_list = list(candidates.keys())
    skill_list = list(project_requirements.keys())
    
    # Создание матрицы стоимости
    cost_matrix = np.zeros((len(candidate_list), len(skill_list)))
    for i, candidate in enumerate(candidate_list):
        for j, skill in enumerate(skill_list):
            candidate_skill_level = candidates[candidate].get(skill, 0)
            required_level = project_requirements[skill]["уровень"]
            if candidate_skill_level >= required_level:
                cost_matrix[i, j] = 1 / (1 + abs(candidate_skill_level - required_level))
            else:
                cost_matrix[i, j] = np.inf  # Неудовлетворение требований
    
    # Добавление фиктивных кандидатов или требований, если размеры не совпадают
    num_candidates, num_skills = cost_matrix.shape
    max_size = max(num_candidates, num_skills)
    cost_matrix = np.pad(cost_matrix, ((0, max_size - num_candidates), (0, max_size - num_skills)), constant_values=np.inf)
    
    # Решение задачи о назначениях
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Формирование результата
    assignments = {}
    selected_candidates = set()
    skill_count = {skill: 0 for skill in skill_list}
    
    for row, col in zip(row_ind, col_ind):
        if row < num_candidates and col < num_skills and cost_matrix[row, col] != np.inf:
            candidate_name = candidate_list[row]
            skill_name = skill_list[col]
            if skill_count[skill_name] < project_requirements[skill_name]["количество"]:
                assignments[candidate_name] = skill_name
                selected_candidates.add(candidate_name)
                skill_count[skill_name] += 1
    
    # Проверка количества исполнителей
    while len(selected_candidates) < Q_min:
        # Добавляем ближайшего по уровню кандидата
        for candidate in candidate_list:
            if candidate not in selected_candidates:
                best_skill = max(candidates[candidate], key=lambda s: candidates[candidate][s], default=None)
                if best_skill and skill_count[best_skill] < project_requirements[best_skill]["количество"]:
                    assignments[candidate] = best_skill
                    selected_candidates.add(candidate)
                    skill_count[best_skill] += 1
                    break
    
    while len(selected_candidates) > Q_max:
        # Удаляем кандидата с наименьшим соответствием
        worst_candidate = min(assignments, key=lambda c: candidates[c][assignments[c]])
        del assignments[worst_candidate]
        selected_candidates.remove(worst_candidate)
    
    return assignments

# Пример входных данных
candidates = {
    "Alice": {"Python": 5, "ML": 4},
    "Bob": {"Python": 3, "ML": 5},
    "Charlie": {"Python": 4, "ML": 3},
    "David": {"Python": 5, "ML": 5}
}

project_requirements = {
    "Python": {"уровень": 4, "количество": 1},
    "ML": {"уровень": 4, "количество": 1}
}

Q_min = 2
Q_max = 3

# Вызов функции и вывод результата
assignments = match_candidates_to_project(candidates, project_requirements, Q_min, Q_max)
print(assignments)