import networkx as nx
import os
from Levenshtein import distance
import json
from tqdm import tqdm
import argparse
import time
import re
from collections import defaultdict

def load_ontology(ontology_path):
    """Загрузка онтологии из GML файла"""
    try:
        if not ontology_path.endswith('.gml'):
            print(f'Ошибка: файл {ontology_path} должен иметь расширение .gml')
            return None
            
        if not os.path.exists(ontology_path):
            print(f'Ошибка: файл {ontology_path} не найден')
            return None
            
        print("Загрузка онтологии...")
        start_time = time.time()
        ontology = nx.read_gml(ontology_path)
        print(f"Онтология загружена за {time.time() - start_time:.2f} секунд")
        return ontology
    except Exception as e:
        print(f'Ошибка при загрузке онтологии: {str(e)}')
        return None

def get_concepts(ontology):
    """Получение списка понятий из онтологии"""
    print("Получение списка понятий...")
    start_time = time.time()
    concepts = list(ontology.nodes())
    print(f"Получено {len(concepts)} понятий за {time.time() - start_time:.2f} секунд")
    return concepts

def find_similar_words(word, text_words, threshold=0.8):
    """Поиск похожих слов с использованием расстояния Левенштейна"""
    similar_words = []
    for text_word in text_words:
        if len(word) > 0 and len(text_word) > 0:
            max_len = max(len(word), len(text_word))
            similarity = 1 - distance(word, text_word) / max_len
            if similarity >= threshold:
                similar_words.append(text_word)
    return similar_words

def load_ontology_entities(json_file):
    """Загрузка терминов из JSON файла"""
    try:
        if not os.path.exists(json_file):
            print(f'Ошибка: файл {json_file} не найден!')
            return []
            
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, dict):
            print(f'Ошибка: файл {json_file} должен содержать словарь!')
            return []
            
        classes = data.get('classes', [])
        if not classes:
            print(f'Ошибка: в файле {json_file} не найдены классы!')
            return []
            
        return classes
    except json.JSONDecodeError:
        print(f'Ошибка: файл {json_file} имеет неправильный формат JSON!')
        return []
    except Exception as e:
        print(f'Ошибка при загрузке файла {json_file}: {str(e)}')
        return []

def extract_ngrams(text, n):
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def analyze_text(text, ontology_terms):
    term_counts = defaultdict(int)
    for term in ontology_terms:
        # Разбиваем термин на слова, если он содержит '_'
        term_words = term.split('_')
        n = len(term_words)
        if n > 1:
            # Для терминов из нескольких слов ищем n-граммы
            ngrams = extract_ngrams(text, n)
            for ngram in ngrams:
                if ngram == ' '.join(term_words):
                    term_counts[term] += 1
        else:
            # Для одиночных слов ищем точное совпадение
            term_counts[term] += text.count(term)
    return term_counts

def process_file(filepath, ontology_terms):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return analyze_text(text, ontology_terms)

def process_folder(folder, ontology_terms):
    results = defaultdict(lambda: defaultdict(int))
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder, filename)
            term_counts = process_file(filepath, ontology_terms)
            for term, count in term_counts.items():
                results[filename][term] = count
            print(f'Обработан файл: {filename}')
    return results

def process_text_files(text_dir, ontology_path, output_file):
    """Обработка всех текстовых файлов"""
    start_time = time.time()
    
    ontology = load_ontology(ontology_path)
    concepts = get_concepts(ontology)
    
    all_results = []
    
    # Получаем список текстовых файлов
    text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
    print(f"\nНайдено {len(text_files)} текстовых файлов для анализа")
    
    # Прогресс по файлам
    for filename in tqdm(text_files, desc="Обработка файлов", leave=True):
        file_path = os.path.join(text_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        file_start_time = time.time()
        results = analyze_text(text, concepts)
        file_time = time.time() - file_start_time
        
        all_results.append(results)
        
        print(f"Файл {filename} обработан за {file_time:.2f} секунд")
        print(f"Найдено {len([r for r in results if r])} частей с упоминаниями понятий")
    
    total_time = time.time() - start_time
    print(f"\nВсего обработано {len(all_results)} файлов")
    print(f"Общее время обработки: {total_time:.2f} секунд")
    print("Сохранение результатов...")
    
    # Сохранение результатов в JSON файл
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Результаты сохранены в файл {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Анализ встречаемости понятий в текстах')
    parser.add_argument('--text_dir', default='data_txt', help='Директория с текстовыми файлами')
    parser.add_argument('--ontology_json', default='processed_ontology_entities.json', help='Путь к файлу с терминами онтологии в формате JSON')
    parser.add_argument('--ontology_gml', default='ml_ontology_graph.gml', help='Путь к файлу онтологии в формате GML')
    parser.add_argument('--output', default='concept_occurrences.json', help='Файл для сохранения результатов')
    
    args = parser.parse_args()
    
    # Загружаем термины из JSON файла
    ontology_terms = load_ontology_entities(args.ontology_json)
    if not ontology_terms:
        print('Нет терминов для анализа! Проверьте файл ontology_entities.json')
        sys.exit(1)
    
    print(f'Загружено {len(ontology_terms)} терминов для анализа')
    
    if not os.path.exists(args.text_dir):
        print(f'Папка {args.text_dir} не найдена!')
        sys.exit(1)
        
    # Анализируем тексты
    results = process_folder(args.text_dir, ontology_terms)
    
    # Сохраняем результаты
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f'\nРезультаты анализа сохранены в файл {args.output}')
    
    # Загружаем и обрабатываем GML онтологию, если файл существует
    if os.path.exists(args.ontology_gml):
        ontology = load_ontology(args.ontology_gml)
        if ontology:
            print(f'Онтология загружена, количество узлов: {len(ontology.nodes())}')
            print(f'Количество связей: {len(ontology.edges())}')

    process_text_files(args.text_dir, args.ontology_gml, args.output) 