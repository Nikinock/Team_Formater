from owlready2 import *
import json

def extract_ontology_entities(ontology_path, output_json_path):
    # Загрузка онтологии
    onto = get_ontology(ontology_path).load()
    
    # Создание словаря для хранения сущностей
    entities = {
        'classes': [],
        'object_properties': [],
        'data_properties': [],
        'individuals': []
    }
    
    # Извлечение классов
    for cls in onto.classes():
        entities['classes'].append(str(cls.name))
    
    # Извлечение объектных свойств
    for prop in onto.object_properties():
        entities['object_properties'].append(str(prop.name))
    
    # Извлечение свойств данных
    for prop in onto.data_properties():
        entities['data_properties'].append(str(prop.name))
    
    # Извлечение индивидов
    for ind in onto.individuals():
        entities['individuals'].append(str(ind.name))
    
    # Сохранение в JSON файл
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    ontology_path = "ml_onto.owl"
    output_json_path = "ontology_entities.json"
    extract_ontology_entities(ontology_path, output_json_path)
