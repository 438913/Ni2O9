# 用type_weight存储态的类型(比如d8_d8_L)和对应的权重,
# d8_d8_L作为key, 权重列表作为value
type_weight = {}
# 读取数据
with open('dL_weight', 'r') as f:
    for line in f:
        if ':' in line and 'ed' not in line:
            dL_type, weight = line.split(':')
            dL_type = dL_type.strip()
            weight = float(weight.strip())
            if dL_type in type_weight:
                type_weight[dL_type] += [weight]
            else:
                type_weight[dL_type] = [weight]

# 计算字典中态的最大维度
max_dim = 0
for weight in type_weight.values():
    dim = len(weight)
    if dim > max_dim:
        max_dim = dim

# 只保留最大维度的类型
type_weight = {key: value for key, value in type_weight.items() if len(value) == max_dim}

# 检验读取数据是否正确
for dL_type, weight in type_weight.items():
    print(dL_type)
    for wgt in weight:
        print(f'\t{wgt}')
