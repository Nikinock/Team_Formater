import json
from path_finder import find_min_path
from collections import defaultdict

MAX_PATH_WEIGHT = 26.3452  # Максимальный вес пути в графе

def load_data(graph_file, input_file):
    """
    Загружает входные данные.
    
    Returns:
        tuple: (students, project_requirements, members_quantity)
    """
    # Загружаем входные данные
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['students'], data['project']['requirements'], data['project']['members_quantity']

def find_closest_skill(graph_file, student_skills, requirement):
    """
    Находит ближайший навык студента к требованию проекта.
    Использует нормализованный вес пути (1 - path_weight/MAX_PATH_WEIGHT).
    
    Args:
        graph_file: путь к файлу графа
        student_skills: список навыков студента в формате [{"skill": level}, ...]
        requirement: требование в формате {"skill": level}
    
    Returns:
        tuple: (closest_skill, normalized_weight, path)
    """
    req_skill = list(requirement.keys())[0]
    min_normalized_weight = 0  # Начальное значение - худший случай
    closest_skill = None
    best_path = None
    
    # Проверяем, есть ли требование среди навыков студента
    for skill_dict in student_skills:
        skill = list(skill_dict.keys())[0]
        if skill == req_skill:
            return skill, 1.0, [skill]  # Точное совпадение - максимальный вес
    
    # Ищем ближайший навык через граф
    for skill_dict in student_skills:
        skill = list(skill_dict.keys())[0]
        try:
            distance, path = find_min_path(graph_file, skill, req_skill)
            # Нормализуем вес: 1 - distance/MAX_PATH_WEIGHT
            normalized_weight = 1 - (distance / MAX_PATH_WEIGHT)
            if normalized_weight > min_normalized_weight:
                min_normalized_weight = normalized_weight
                closest_skill = skill
                best_path = path
        except (ValueError, FileNotFoundError):
            continue  # Пропускаем, если путь не найден
    
    return closest_skill, min_normalized_weight, best_path

def calculate_compatibility_vector(graph_file, student, requirements):
    """
    Вычисляет вектор соответствия студента требованиям проекта.
    Каждый элемент вектора - это нормализованный вес пути до ближайшего навыка.
    
    Args:
        graph_file: путь к файлу графа
        student: словарь с данными студента
        requirements: список требований проекта
    
    Returns:
        tuple: (compatibility_vector, skill_matches)
            compatibility_vector: список нормализованных весов для каждого требования
            skill_matches: список соответствий (требование -> навык студента)
    """
    student_skills = student['skills']
    compatibility_vector = []
    skill_matches = []
    
    for req in requirements:
        req_skill = list(req.keys())[0]
        closest_skill, normalized_weight, path = find_closest_skill(graph_file, student_skills, req)
        
        # Добавляем нормализованный вес в вектор соответствия
        compatibility_vector.append(normalized_weight)
        
        skill_matches.append({
            'requirement': req_skill,
            'matched_skill': closest_skill,
            'normalized_weight': normalized_weight,
            'path': path
        })
    
    return compatibility_vector, skill_matches

def match_skills(graph_file, input_file):
    """
    Основная функция для сопоставления навыков студентов с требованиями проекта.
    
    Returns:
        dict: результаты сопоставления для каждого студента
    """
    students, requirements, members_quantity = load_data(graph_file, input_file)
    results = {}
    
    print(f"\nАнализ соответствия {len(students)} студентов требованиям проекта:")
    print("Требуемые навыки:")
    req_skills = []
    for req in requirements:
        skill = list(req.keys())[0]
        req_skills.append(skill)
        print(f"  {skill}")
    
    print("\nРезультаты:")
    for student in students:
        name = f"{student['meta']['name']} {student['meta']['surname']}"
        compatibility_vector, _ = calculate_compatibility_vector(graph_file, student, requirements)
        
        # Формируем строку с нормализованными весами
        weights_str = "; ".join([f"{req}: {weight:.4f}" for req, weight in zip(req_skills, compatibility_vector)])
        print(f"{name}[{weights_str}]")
        
        results[student['meta']['e-mail']] = {
            'name': name,
            'compatibility_vector': compatibility_vector
        }
    print(results)
    return results

if __name__ == "__main__":
    try:
        match_skills("graph_weights.json", "input_data.json")
    except FileNotFoundError as e:
        print(f"Ошибка: Файл не найден - {e}")
    except json.JSONDecodeError:
        print("Ошибка: Неверный формат JSON файла")
    except Exception as e:
        print(f"Произошла ошибка: {e}") 