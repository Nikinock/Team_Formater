from owlready2 import *
import networkx as nx
import matplotlib.pyplot as plt
import os

def get_safe_name(cls):
    """Безопасное получение имени класса"""
    try:
        if hasattr(cls, 'name'):
            return str(cls.name)
        elif hasattr(cls, 'label'):
            return str(cls.label[0])
        else:
            return str(cls).split('.')[-1]
    except:
        return str(cls)

def create_ml_graph():
    # Создание графа
    G = nx.DiGraph()
    
    # Загрузка онтологии
    onto = get_ontology("ml_onto.owl").load()
    
    # Создание узлов для классов
    for cls in onto.classes():
        class_name = get_safe_name(cls)
        G.add_node(class_name, 
                  type='Class',
                  iri=str(cls.iri))
    
    # Создание узлов для свойств и отношений
    for prop in onto.properties():
        prop_name = get_safe_name(prop)
        G.add_node(prop_name,
                  type='Property',
                  iri=str(prop.iri))
        
        # Создание отношений между свойствами и классами
        if prop.domain:
            for domain in prop.domain:
                domain_name = get_safe_name(domain)
                if not G.has_node(domain_name):
                    G.add_node(domain_name, type='Class')
                G.add_edge(domain_name, prop_name, 
                          relation='HAS_PROPERTY')
        
        if prop.range:
            for range_cls in prop.range:
                # Проверяем, является ли range_cls классом онтологии
                if hasattr(range_cls, 'name') or hasattr(range_cls, 'label'):
                    range_name = get_safe_name(range_cls)
                    if not G.has_node(range_name):
                        G.add_node(range_name, type='Class')
                    G.add_edge(prop_name, range_name, 
                              relation='RANGE')
                else:
                    # Если это не класс онтологии, создаем узел с типом значения
                    value_type = str(range_cls.__name__)
                    G.add_node(value_type, type='ValueType')
                    G.add_edge(prop_name, value_type, 
                              relation='RANGE')
    
    # Создание отношений наследования
    for cls in onto.classes():
        for parent in cls.is_a:
            if isinstance(parent, ThingClass):
                child_name = get_safe_name(cls)
                parent_name = get_safe_name(parent)
                if not G.has_node(parent_name):
                    G.add_node(parent_name, type='Class')
                G.add_edge(child_name, parent_name, 
                          relation='IS_A')
    
    # Визуализация графа
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Рисуем узлы разных типов разными цветами
    class_nodes = [node for node, data in G.nodes(data=True) if data.get('type') == 'Class']
    property_nodes = [node for node, data in G.nodes(data=True) if data.get('type') == 'Property']
    value_type_nodes = [node for node, data in G.nodes(data=True) if data.get('type') == 'ValueType']
    
    nx.draw_networkx_nodes(G, pos, nodelist=class_nodes, node_color='lightblue', 
                          node_size=2000, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=property_nodes, node_color='lightgreen', 
                          node_size=2000, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=value_type_nodes, node_color='lightpink', 
                          node_size=2000, alpha=0.7)
    
    # Рисуем рёбра
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    
    # Добавляем метки
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Сохраняем граф
    plt.savefig('ml_ontology_graph.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохраняем граф в формате GML для дальнейшего использования
    nx.write_gml(G, 'ml_ontology_graph.gml')
    
    print("Граф успешно создан и сохранен в файлы 'ml_ontology_graph.png' и 'ml_ontology_graph.gml'")

if __name__ == "__main__":
    create_ml_graph()
