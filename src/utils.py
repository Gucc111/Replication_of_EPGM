import numpy as np

# __all__ = ['attr_matrix']

def attr_matrix(attr_list, case_list):
    result = []
    
    for case in case_list:
        result.append([case.all_attrs[attr_name]['value'] for attr_name in attr_list])
    
    return np.array(result, dtype='object')

def get_knowledge(case_list, knowledge='scenario'):
    results = []
    
    for case in case_list:
        if knowledge == 'scenario':
            results.append(case.scenario)
        elif knowledge == 'effect':
            results.append(case.effect)
        else:
            results.append(case.action)
    
    return results