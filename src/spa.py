import numpy as np
from .utils import *

# 得到属性的最值
def get_extreme_value(matrix, cost=True):
    if cost:
        best_matrix = matrix.min(axis=0)
        worst_matrix = matrix.max(axis=0)
    else:
        best_matrix = matrix.max(axis=0)
        worst_matrix = matrix.min(axis=0)
    
    return best_matrix, worst_matrix

# 熵权法
def entropy_weight(matrix):
    eps = 1e-12  # 防止除0或log(0)
    
    if isinstance(matrix, list):
        indice = matrix[0].shape[1]

        weights = entropy_weight(np.concatenate(matrix, axis=1))

        weights_a = weights[:indice].sum()
        weights_b = weights[indice:].sum()

        return np.array([weights_a, weights_b]) # 一级权重
    else:
        m, _ = matrix.shape
        
        col_sum = matrix.sum(axis=0, keepdims=True)
        col_sum = np.where(col_sum == 0, eps, col_sum)
        p = matrix / col_sum
        
        term = 1.0 / np.log(m)
        
        p_log_p = np.where(p != 0, p * np.log(p), 0)
        
        e = -term * np.sum(p_log_p, axis=0)
        d = 1 - e
        
        weights = d / np.sum(d)
    
    return weights # 二级权重

# 联系度
def compute_association(matrix, best_matrix, worst_matrix, cost=True):
    a = (best_matrix * worst_matrix) / ((best_matrix + worst_matrix) * matrix)
    c = matrix / (best_matrix + worst_matrix)
    
    if cost: # 成本型属性矩阵
        v = a / (a + c)
        return v.min(axis=0) / v
    else:
        v = c / (a + c)
        return v / v.max(axis=0)

# 集对分析法
def set_pari_analysis(case_list):
    effect_list = get_knowledge(case_list, 'effect') # 抽取效果知识元
    # 两类属性名
    attr_cost = case_list[0].cost_attr
    attr_benefit = case_list[0].benefit_attr
    # 效用矩阵
    matrix_cost = (attr_matrix(attr_cost, effect_list)).astype(float)
    matrix_benefit = (attr_matrix(attr_benefit, effect_list)).astype(float)
    
    # 最值
    best_u1, worst_u1 = get_extreme_value(matrix_cost)
    best_u2, worst_u2 = get_extreme_value(matrix_benefit, cost=False)

    weights_cost = entropy_weight(matrix_cost)
    weights_benefit = entropy_weight(matrix_benefit)
    weights_first = entropy_weight([matrix_cost, matrix_benefit]) # 一级权重

    # 两类集对贴近度
    v1 = compute_association(matrix_cost, best_u1, worst_u1)
    v2 = compute_association(matrix_benefit, best_u2, worst_u2, cost=False)

    r1 = (weights_cost * v1).sum(axis=1)
    r2 = (weights_benefit * v2).sum(axis=1)
    r = np.tile([r1, r2], reps=1)

    # 综合评价值
    r = weights_first.reshape(1, 2) @ r

    return case_list[r.argmax()] # 返回评价最高的案例作为初步应急方案