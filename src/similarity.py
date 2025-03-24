import numpy as np
from .utils import *

# 概念相似度
def sim_concept(ec_case_list, object_case, alpha):
    results = []
    
    for ec_case in ec_case_list:
        a = set(ec_case.concepts)
        b = set(object_case.concepts)
        if a <= b or b <= a: # 若其中一个集合为另一个集合的子集，则返回1
            results.append(1)
        else:
            numerator = len(a & b)
            
            term1 = alpha * len(a - b)
            term2 = (1 - alpha) * len(b - a)
            denominator = numerator + term1 + term2
            
            results.append(numerator / denominator)
    
    return np.array(results).reshape(-1, 1)

# 结构相似度
def sim_structure(ec_case_list, object_case):
    results = []
    num_object_attrs = len(object_case.attr_names)
    
    for ec_case in ec_case_list:
        num_ec_attrs = len(ec_case.attr_names)
        
        numerator = len(set(ec_case.attr_names) & set(object_case.attr_names))
        
        denominator = num_ec_attrs + num_object_attrs - numerator
        
        results.append(numerator / denominator)
    
    return np.array(results)

# 确定符号型属性值相似度
def sim_attr_1(attr_ec, attr_object):
    return (attr_ec == attr_object).astype(float).reshape(-1, 1)

# 确定数值型属性值相似度
def sim_attr_2(attr_ec, attr_object, unit):
    numerator = np.abs(attr_ec - attr_object)
    
    if isinstance(unit[0], str): # 判断是否为离散型
        temp = np.concatenate([attr_ec, attr_object]).astype(float)
        # 在取小或取大前替换掉缺失值
        upper = np.nan_to_num(temp, nan=-np.inf).max()
        lower = np.nan_to_num(temp, nan=np.inf).min()
    else:
        upper = max(unit)
        lower = min(unit)
    
    denominator = upper - lower
    
    return (1 - numerator / denominator).astype(float).reshape(-1, 1)

# 三角模糊数
def fuzzy_num(p, n):
    # 计算模糊数上下界时对缺失值进行处理
    temp = np.nan_to_num((p - 1) / n, nan=np.inf)
    temp = np.maximum(temp, 0)
    q_l = np.where(np.isinf(temp), np.nan, temp).reshape(-1, 1)
    
    q_m = (p / n).reshape(-1, 1)
    
    temp = np.nan_to_num((p + 1) / n, nan=-np.inf)
    temp = np.minimum(temp, 1)
    q_u = np.where(np.isneginf(temp), np.nan, temp).reshape(-1, 1)
    
    return np.concatenate([q_l, q_m, q_u], axis=1)

# 模糊概念型属性值相似度
def sim_attr_3(attr_ec, attr_object, unit):
    n = len(unit) - 1
    p_ec = np.array([unit.index(x) if x in unit else None for x in attr_ec], dtype=float)
    p_object = np.array([unit.index(x) if x in unit else None for x in attr_object], dtype=float)
    
    fz_ec = fuzzy_num(p_ec, n)
    fz_object = fuzzy_num(p_object, n)
    
    return (1 - np.abs(fz_ec - fz_object).mean(axis=1)).reshape(-1, 1)

# 取出区间型属性值的上下界
def get_num(x, index='lower'):
    if index == 'lower':
        return x[0]
    else:
        return x[1]

vec_get_num = np.vectorize(get_num)

# 符号函数
def sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

vec_sgn = np.vectorize(sgn)

# 区间型属性值相似度
def sim_attr_4(attr_ec, attr_object):
    q_array = np.concatenate([vec_get_num(attr_ec, 'lower').reshape(-1, 1), vec_get_num(attr_ec, 'upper').reshape(-1, 1)], axis=1)
    
    # 应急案例和目标问题的区间属性的上下界
    ul_ec = q_array
    ul_object = np.concatenate([vec_get_num(attr_object, 'lower'), vec_get_num(attr_object, 'upper')]).reshape(1, -1).repeat(10, axis=0)
    
    q_array = np.concatenate([q_array, ul_object], axis=1)
    q_array = np.sort(q_array, axis=1) # 对应急案例和目标问题的区间属性的上下界进行排序
    
    diff_14 = q_array[:, -1] - q_array[:, 0]
    diff_23 = q_array[:, -2] - q_array[:, 1]
    
    yl_xu = vec_sgn(ul_object[:, 0] - ul_ec[:, 1])
    xl_yu = vec_sgn(ul_ec[:, 0] - ul_object[:, 1])
    
    numerator = diff_23 * (1 - yl_xu) * (1 - xl_yu) / 4
    denominator = diff_14 - diff_23 * np.abs(yl_xu - xl_yu) / 2
    
    return (numerator / denominator).reshape(-1, 1)

def sim_attr(case_list, object_scenario):
    common_attrs = object_scenario.get_common_attrs(case_list)
    measure_type = object_scenario.get_attr_dtype(common_attrs)
    unit_list = object_scenario.get_unit(common_attrs)

    matrix_ec = attr_matrix(common_attrs, case_list)
    matrix_object = attr_matrix(common_attrs, [object_scenario])

    results = []
    
    for i, j in enumerate(measure_type):
        if j == 1: # 确定符号型
            results.append(sim_attr_1(matrix_ec[:, i], matrix_object[:, i]))
        elif j == 2: # 确定数值型
            results.append(sim_attr_2(matrix_ec[:, i], matrix_object[:, i], unit_list[i]))
        elif j == 3: # 模糊概念型
            results.append(sim_attr_3(matrix_ec[:, i], matrix_object[:, i], unit_list[i]))
        else: # 区间数型
            results.append(sim_attr_4(matrix_ec[:, i], matrix_object[:, i]))
    
    return np.concatenate(results, axis=1)

# 综合相似度
def sim_scen(case_list, object_scenario, alpha, tao, weights=None):
    similarity_concepts = sim_concept(case_list, object_scenario, alpha)
    indices = np.where(similarity_concepts >= tao)[0]
    case_list = [case_list[i] for i in indices]

    similarity_structure = sim_structure(case_list, object_scenario)

    similarity_attributes = sim_attr(case_list, object_scenario)
    
    if not weights:
        similarity_scenario = similarity_structure * np.nanmean(similarity_attributes, axis=1)
        return similarity_scenario
    else:
        weights = np.array(weights)
        similarity_scenario = similarity_structure * np.nanmean(weights * similarity_attributes, axis=1)
        return similarity_scenario

# 得到备选方案集
def get_alternative(case_list, object_case, alpha, tao):
    scenario_list = get_knowledge(case_list)
    
    similarity_scenario = sim_scen(scenario_list, object_case.scenario, alpha, tao)
    indices = np.where(similarity_scenario > tao)[0]
    
    alternative_list = []
    for i in indices:
        alternative_list.append(case_list[i])
    
    return alternative_list