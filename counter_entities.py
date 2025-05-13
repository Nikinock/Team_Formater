import os
import glob
import json

with open("final_ontology_entities.json", "r", encoding="utf-8") as f:
    entities = json.load(f)

folder = "data_processed"
results = []

for file_index, filepath in enumerate(glob.glob(os.path.join(folder, "*.txt")), start=1):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    parts = content.split('\n\n')  # делим на части по пустой строке
    part_results = []

    for part_index, part in enumerate(parts, start=1):
        words = part.lower().split()
        term_counts = {}

        for entiti in entities['classes']:
            ent_words = entiti.lower().split()
            ent_len = len(ent_words)
            count = 0

            for i in range(len(words) - ent_len + 1):
                if words[i:i + ent_len] == ent_words:
                    count += 1

            if count > 0:
                term_counts[entiti] = count

        if term_counts:
            part_results.append({
                "part": part_index,
                "terms": term_counts
            })

    if part_results:
        results.append({
            "text": file_index,
            "parts": part_results
        })

with open("entity_counts_by_parts.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)