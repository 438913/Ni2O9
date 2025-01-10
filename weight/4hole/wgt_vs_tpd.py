import matplotlib.pyplot as plt
import numpy as np

# 用type_weight存储态的类型(比如d8_d8_L)和对应的权重,
# 态的类型作为key, 权重列表作为value
dL_type_list = ('d8_d8', 'd8_d9L', 'd8_O_d9','d9L_d9L',
                'L2_d8', 'd9_O_d9L', 'L_O_d8')
type_num = len(dL_type_list)
tpds = np.linspace(0.79, 1.738, num=11, endpoint=True)
tpd_num = len(tpds)
dft_tpd = 1.58
type_weight = np.zeros((type_num, tpd_num))
# 读取数据
with open('weight_vs_tpd', 'r') as f:
    i = 0
    for line in f:
        if 'tpd' in line:
            i += 1
        if ':' in line and 'ed' not in line:
            dL_type, weight = line.split(':')
            dL_type = dL_type.strip()
            weight = float(weight.strip())
            if dL_type in dL_type_list:
                idx = dL_type_list.index(dL_type)
                type_weight[idx][i - 1] = weight

# 检验读取数据是否正确
for i in range(type_num):
    print(dL_type_list[i])
    for j in range(tpd_num):
        print(type_weight[i][j])

# 主图
fig, ax = plt.subplots(figsize=(7.8, 8))
labels = (r'$d^8$-$d^8$', r'$d^8$-$d^9L$', r'$d^8$-$O$-$d^9$', r'$d^9L$_$d^9L$',
          r'$d^8$_$L^2$', r'$d^9$-$O$-$d^9L$', r'$d8$_$O$_$L$')
line_styles = ('g', 'peru', 'cyan', 'k', 'y', 'r', 'm', 'orange')
marker_styles = ('d', '^', 'o', 'p', '>', '|', 'x', '*')
for i in range(type_num):
    ax.plot(tpds, type_weight[i], line_styles[i], label=labels[i], marker=marker_styles[i], markersize=10)
ax.plot(dft_tpd, 0.005, '*', color='r', markersize=25)
ax.set_xlabel(r'$t_{pd}$', fontsize=30)
ax.set_ylabel('Weight', fontsize=30)
ax.set_xlim([0, 4.2])
ax.set_ylim([0, 1.2])
ax.tick_params(axis='both', labelsize=24)
ax.legend(loc='best', bbox_to_anchor=(0.50, 0.30, 0.5, 0.5), fontsize=16, framealpha=0.5, edgecolor='black')
fig.tight_layout()
fig.show()

# 子图
