class AttrStatesKnowledge:
    def __init__(self):
        self.attributes = {}
    
    # 添加属性
    def add_attribute(self, attr_name, value, dtype=None, unit=None):
        '''
        情景知识元属性 dtype
        0: 不可测
        1: 确定符号型
        2: 确定数值型
        3: 模糊概念型
        4: 区间数型属性
        效果知识元属性dype
        0: 成本型
        1: 效益型
        '''
        if attr_name in self.attributes:
            raise ValueError(f"属性 '{attr_name}' 已存在，禁止重复添加！")
        self.attributes[attr_name] = {'value': value,
                                      'dtype': dtype,
                                      'unit': unit}
    
    # 批量添加属性
    def add_attributes(self, attr_list):
        if isinstance(attr_list[0], list):
            for item in attr_list:
                attr_name, value, dtype, unit = item
                self.add_attribute(attr_name, value, dtype=dtype, unit=unit)
        else:
            attr_name, value, dtype, unit = attr_list
            self.add_attribute(attr_name, value, dtype=dtype, unit=unit)
    
    # 删除属性
    def delete_attribute(self, attr_name):
        if attr_name not in self.attributes:
            raise KeyError(f"属性 '{attr_name}' 不存在，无法删除！")
        del self.attributes[attr_name]
    
    # 修改属性
    def update_attribute(self, attr_name, value=None, dtype=None, unit=None):
        if attr_name not in self.attributes:
            raise KeyError(f"属性 '{attr_name}' 不存在，无法更新！")
        
        if value is not None:
            self.attributes[attr_name]['value'] = value
        
        if dtype is not None:
            self.attributes[attr_name]['dtype'] = dtype
        
        if unit is not None:
            self.attributes[attr_name]['unit'] = unit
    
    # 获取属性名
    def get_attrnames(self):
        return list(self.attributes.keys())
    
    # 获取属性取值
    def get_attrvalues(self):
        value_list = []
        for attr_desc in self.attributes.values():
            value_list.append(attr_desc['value'])
        return value_list
    
    # 获取属性可测特征或效益类型
    def get_attr_dtype(self, attr_name):
        if isinstance(attr_name, list):
            dtype_list = []
            for attr in attr_name:
                dtype_list.append(self.attributes[attr]['dtype'])
            return dtype_list
        else:
            return self.attributes[attr_name]['dtype']
    
    def __repr__(self):
        return f"AttributeStates(attributes={self.attributes})"

class RelationKnowledge:
    def __init__(self):
        self.relations = []
    # 添加关系
    def add_relation(self, relation_type, input, output, func=None):
        rel = {"relation_type": relation_type,
               "input_attrs": input,
               "output_attrs": output,
               "func": func}
        self.relations.append(rel)
    # 批量添加关系
    def add_relations(self, relation_list):
        if isinstance(relation_list[0], list):
            for item in relation_list:
                relation_type, input, output, func = item
                self.add_relation(relation_type, input, output, func)
        else:
            relation_type, input, output, func = relation_list
            self.add_relation(relation_type, input, output, func)
    
    def __repr__(self):
        return f"RelationKnowledge(relations={self.relations})"

class KnowledgeElement:
    def __init__(self, concepts=None):
        self.concepts = concepts
        self.attrstates = AttrStatesKnowledge()
        self.attrrelations = RelationKnowledge()
    
    # 所有属性名
    @property
    def attr_names(self):
        return self.attrstates.get_attrnames()
    
    # 所有属性状态
    @property
    def all_attrs(self):
        return self.attrstates.attributes
    
    # 所有属性的取值
    @property
    def all_attrvalues(self):
        return self.attrstates.get_attrvalues()
    
    def add_attribute(self, attr_name, value, dtype=None, unit=None):
        self.attrstates.add_attribute(attr_name, value, dtype, unit)
    
    def add_attributes(self, attr_list):
        self.attrstates.add_attributes(attr_list)
    
    def update_attribute(self, attr_name, value=None, dtype=None, unit=None):
        self.attrstates.update_attribute(attr_name, value, dtype, unit)
    
    def get_attrvalues(self):
        return self.attrstates.get_attrvalues()
    
    def get_attr_dtype(self, attr_name):
        return self.attrstates.get_attr_dtype(attr_name)
    
    def get_common_attrs(self, case_list):
        common_attributes = set(self.attr_names)
        for knowledge in case_list:
            common_attributes = common_attributes & set(knowledge.attr_names)
        return list(common_attributes)
    
    # 获取unit
    def get_unit(self, attr_name):
        if isinstance(attr_name, list):
            unit_list = []
            for attr in attr_name:
                unit_list.append(self[attr]['unit'])
            return unit_list
        else:
            return self[attr_name]['unit']
    
    def __getitem__(self, index):
        return self.attrstates.attributes[index]

class Case():
    def __init__(self, attr_scenario, attr_effect, concept=None, case=True):
        self.scenario = KnowledgeElement(concept)
        self.scenario.add_attributes(attr_scenario)
        
        if case:
            self.effect = KnowledgeElement()
            self.effect.add_attributes(attr_effect)
           
            self.cost_attr = []
            self.benefit_attr = []
           
            for attr in self.effect.attr_names:
                if self.effect.get_attr_dtype(attr) == 0:
                    self.cost_attr.append(attr)
                else:
                    self.benefit_attr.append(attr)
            
            self.action = KnowledgeElement()
            # 后续补充