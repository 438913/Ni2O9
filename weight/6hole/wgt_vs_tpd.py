import matplotlib.pyplot as plt
import numpy as np

# 用type_weight存储态的类型(比如d8_d8_L)和对应的权重,
# 态的类型作为key, 权重列表作为value
dL_type_list = ('d8L_d8L', 'd7_d8L', 'd8L_d9L2', 'd8L_O_d9L',
                'd7_d7', 'd8_O_d8L', 'd7_O_d9L')
type_num = len(dL_type_list)
tpds = np.linspace(0.79, 1.738, num=11, endpoint=True)
tpd_num = len(tpds)
dft_tpd = 1.58
type_weight = np.zeros((type_num, tpd_num))     # 行为dL类型，列为tpd, 元素为权重

# 读取数据
with open('dL_weight', 'r') as f:
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
labels = (r'$d^8L$-$d^8L$', r'$d^7$-$d^8L$', r'$d^8L$-$d^9L^2$', r'$d^8L$-$O$-$d^9L$',
          r'$d^7$-$d^7$', r'$d^8$-$O$-$d^8L$', r'$d^7$-$O$-$d^9L$')
line_styles = ['g', 'k', 'peru', 'orange', 'y', 'r', 'deeppink']
marker_styles = ['d', 'D', 'D', '*', '>', '+', 'x']

for i in range(type_num):
    ax.plot(tpds, type_weight[i], line_styles[i], label=labels[i], marker=marker_styles[i], markersize=10)
ax.plot(dft_tpd, 0.005, '*', color='r', markersize=25)
ax.set_xlabel(r'$t_{pd}$', fontsize=30)
ax.set_ylabel('Weight', fontsize=30)
ax.set_xlim([0.79, 1.738])
ax.set_ylim([0, 1.3])
ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.tick_params(axis='both', labelsize=24)
ax.legend(loc='best', bbox_to_anchor=(0.50, 0.175, 0.51, 0.5), fontsize=16, framealpha=0.5, edgecolor='black')
ax.text(0.67, 1.3, '(b)', fontsize=30)      # 在图的左上角添加(b)
fig.tight_layout()

# 子图
ax1 = fig.add_axes((0.26, 0.55, 0.24, 0.37))
upupdn_dndnup = []
upupdn_updndn = []
dnupdn_upupdn = []
dnupdn_updnup = []

with open('max_orb_weight', 'r') as f:
    for line in f:
        if '1th total weight' in line:
            weight = line.split('=')[-1].strip()
            weight = float(weight)
            upupdn_dndnup.append(weight)
        if '2th total weight' in line:
            weight = line.split('=')[-1].strip()
            weight = float(weight)
            upupdn_updndn.append(weight)
        if '3th total weight' in line:
            weight = line.split('=')[-1].strip()
            weight = float(weight)
            dnupdn_upupdn.append(weight)
        if '4th total weight' in line:
            weight = line.split('=')[-1].strip()
            weight = float(weight)
            dnupdn_updnup.append(weight)

label_style = ("$d_{z^2}^{\\uparrow}[d_{x^2}^{\\uparrow}L^{\\downarrow}]$-$d_{z^2}^{\\downarrow}$[$d_{x^2}^{\\downarrow}L^{\\uparrow}$](B)",
              "$d_{z^2}^{\\uparrow}$[$d_{x^2}^{\\uparrow}L^{\\downarrow}$]-$d_{z^2}^{\\uparrow}d_{x^2}^{\\downarrow}L^{\\downarrow}$(B)",
               "$d_{z^2}^{\\downarrow}d_{x^2}^{\\uparrow\\downarrow}$-$d_{z^2}^{\\uparrow}$[$d_{x^2}^{\\uparrow}L^{\\downarrow}$](B)",
               "$d_{z^2}^{\\downarrow}d_{x^2}^{\\uparrow\\downarrow}$-$d_{z^2}^{\\uparrow}$$d_{x^2}^{\\downarrow}L^{\\uparrow}$(B)")

ax1.plot(tpds, dnupdn_upupdn, 'r', marker='*', label=label_style[2], markersize=6)
ax1.plot(tpds, upupdn_updndn, 'cyan', marker='x', label=label_style[1], markersize=6)
ax1.plot(tpds, upupdn_dndnup, 'pink', marker='>', label=label_style[0], markersize=7)
ax1.plot(tpds, dnupdn_updnup, 'purple', marker='v', label=label_style[3], markersize=6)

ax1.legend(loc='best', bbox_to_anchor=(0.98, 0.9, 0.1, 0.1), fontsize=16, framealpha=0.5, edgecolor='black')
ax1.plot(dft_tpd, 0.052, '*', color='r', markersize=20)
ax1.set_ylim(0.05, 0.2)
ax1.set_xticks([0.8, 1.2, 1.6])
ax1.tick_params(labelsize=20)

fig.show()
fig.savefig('fig7b.pdf')
