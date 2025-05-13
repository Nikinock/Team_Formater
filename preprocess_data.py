import os
import re

def process_file(filepath, n_sentences=5):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    # Удаляем всё до слова 'introduction' (включая его) и после 'references'
    intro_match = re.search(r'\bintroduction\b', text)
    ref_match = re.search(r'\breferences\b', text)
    
    if intro_match:
        start = intro_match.end()
    else:
        start = 0
    if ref_match:
        end = ref_match.start()
    else:
        end = len(text)
    
    text = text[start:end].strip()
    
    # Удаляем ссылки
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Удаляем абзацы, начинающиеся со слов 'figure (число):' или 'table (число):' и заканчивающиеся двойным переносом строки
    text = re.sub(r'figure\s*\(\d+\):.*?\n\n', '', text, flags=re.DOTALL)
    text = re.sub(r'table\s*\(\d+\):.*?\n\n', '', text, flags=re.DOTALL)
    
    # Удаляем конструкции с числами из нескольких цифр и точкой
    text = re.sub(r'\b\d+\.', '', text)
    
    # Оставляем только буквы, пробелы, переносы, точки и знак '-'
    text = re.sub(r'[^a-z\s\n\.-]', '', text)
    
    # Удаляем одиночные буквы
    text = re.sub(r'\b[a-z]\b', '', text)
    
    # Объединяем перенесенные слова (слова, оканчивающиеся на '-')
    text = re.sub(r'(\w+)-\s*\n*\s*(\w+)', r'\1\2', text)
    
    # Удаляем все переносы строк
    text = text.replace('\n', ' ')

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'\s.\s', '', text)

    # Удаляем дефисы, если они рядом с пробелами или переносами строк
    text = re.sub(r'(?<=\s)-|-(?=\s)|(?<=\n)-|-(?=\n)', '', text)
    
    # Разделяем текст на части по n предложений
    sentences = re.split(r'\.\s+', text)
    parts = []
    for i in range(0, len(sentences), n_sentences):
        part = '. '.join(sentences[i:i+n_sentences])
        if part:
            parts.append(part)
    
    new_text = '\n\n'.join(parts)
    
    filename = os.path.basename(filepath)
    new_filepath = os.path.join('data_processed', filename)

    # Сохраняем результат в новый файл
    with open(new_filepath, 'w', encoding='utf-8') as f:
        f.write(new_text)

def process_folder(folder, n_sentences=5):
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder, filename)
            process_file(filepath, n_sentences)
            print(f'Обработан файл: {filename}')

if __name__ == "__main__":
    import sys
    folder = 'data'
    n_sentences = 5
    if len(sys.argv) > 1:
        try:
            n_sentences = int(sys.argv[1])
        except ValueError:
            print('Ошибка: количество предложений должно быть числом')
            sys.exit(1)
    
    if not os.path.exists(folder):
        print(f'Папка {folder} не найдена!')
    else:
        process_folder(folder, n_sentences)
        print('Обработка завершена.')
