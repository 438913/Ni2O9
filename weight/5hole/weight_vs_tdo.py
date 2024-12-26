import matplotlib.pyplot as plt
import numpy as np

# 用type_weight存储态的类型(比如d8_d8_L)和对应的权重,
# 态的类型作为key, 权重列表作为value
dL_type_list = ('d8_d8L', 'd7_d8', 'd8_O_d8', 'd8_O_d9L',
                'd8L_d9L', 'd8_d9L2', 'd7_d9L', 'd9L_d9L2')
type_num = len(dL_type_list)
tdos = np.linspace(0, 4.2, num=22, endpoint=True)
tdo_num = len(tdos)
dft_tdo = 1.66
type_weight = np.zeros((type_num, tdo_num))
# 读取数据
with open('weight_vs_tdo', 'r') as f:
    i = 0
    for line in f:
        if 'tdo' in line:
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
    for j in range(tdo_num):
        print(type_weight[i][j])

# 主图
fig, ax = plt.subplots(figsize=(7.8, 8))
labels = (r'$d^8$-$d^8L$', r'$d^7$-$d^8$', r'$d^8$-$O$-$d^8$', r'$d^8$-$O$-$d^9L$',
          r'$d^8L$-$d^9L$', r'$d^8$-$d^9L^2$', r'$d^7$-$d^9L$', r'$d^9L$-$d^9L^2$')
line_styles = ['g', 'peru', 'cyan', 'k', 'y', 'r', 'm', 'orange']
marker_styles = ['d', '^', 'o', 'p', '>', '|', 'x', '*']
tdos1 = np.arange(0, 1.41, 0.2)
length1 = len(tdos1)
tdos2 = np.arange(1.6, 4.21, 0.2)
for i in range(type_num):
    ax.plot(tdos1, type_weight[i][: length1], line_styles[i], label=labels[i], marker=marker_styles[i], markersize=10)
    ax.plot(tdos2, type_weight[i][length1:], line_styles[i], marker=marker_styles[i], markersize=10)
ax.axvline(1.5, ymax=0.4, color='k', linewidth=0.8)
ax.plot(dft_tdo, 0.005, '*', color='r', markersize=25)
ax.set_xlabel(r'$t_{do}$', fontsize=30)
ax.set_ylabel('Weight', fontsize=30)
ax.set_xlim([0, 4.2])
ax.set_ylim([0, 1.2])
ax.tick_params(axis='both', labelsize=24)
ax.legend(loc='best', bbox_to_anchor=(0.50, 0.30, 0.5, 0.5), fontsize=16, framealpha=0.5, edgecolor='black')
fig.tight_layout()
fig.show()
