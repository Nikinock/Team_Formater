import json
import re

def process_terms(input_file, output_file):
    """Обработка терминов: приведение к нижнему регистру и замена разделителей"""
    try:
        # Чтение исходного файла
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Проверка структуры данных
        if not isinstance(data, dict) or 'classes' not in data:
            print('Ошибка: файл должен содержать словарь с ключом "classes"')
            return False
            
        # Обработка терминов
        processed_terms = []
        for term in data['classes']:
            # Приведение к нижнему регистру
            term = term.lower()
            # Удаляем подчёркивания
            term = re.sub(r'_', ' ', term)
            # Удаляем дефисы, если они рядом с пробелами или переносами строк
            term = re.sub(r'(?<=\s)-|-(?=\s)|(?<=\n)-|-(?=\n)', '', term)

            processed_terms.append(term)
        
        # Создание нового словаря с обработанными терминами
        processed_data = {'classes': processed_terms}
        
        # Сохранение в новый файл
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
        print(f'Обработано {len(processed_terms)} терминов')
        print(f'Результат сохранен в файл {output_file}')
        return True
        
    except json.JSONDecodeError:
        print('Ошибка: файл имеет неправильный формат JSON')
        return False
    except Exception as e:
        print(f'Произошла ошибка: {str(e)}')
        return False

if __name__ == "__main__":
    input_file = 'ontology_entities.json'
    output_file = 'processed_ontology_entities.json'
    
    if process_terms(input_file, output_file):
        print('Обработка завершена успешно')
    else:
        print('Обработка завершена с ошибками') 