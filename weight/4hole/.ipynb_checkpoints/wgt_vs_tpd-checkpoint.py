import matplotlib.pyplot as plt
import numpy as np

# 数据采用除了tpd和tdo之外，其他参数均使用29.5GPa的DFT数据，tdo = 1.05 tpd

# 用type_weight存储态的类型(比如d8_d8_L)和对应的权重,
# 态的类型作为key, 权重列表作为value
dL_type_list = ('d8_d8', 'd8_d9L', 'd8_O_d9', 'd9L_d9L', 'd9_O_d9L')
type_num = len(dL_type_list)
tpds = np.linspace(0.79, 1.738, num=11, endpoint=True)
tpd_num = len(tpds)
dft_tpd = 1.58
type_weight = np.zeros((type_num, tpd_num))
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
# for i in range(type_num):
#     print(dL_type_list[i])
#     for j in range(tpd_num):
#         print(type_weight[i][j])

# 主图
fig, ax = plt.subplots(figsize=(7.8, 8))
labels = (r'$d^8$-$d^8$', r'$d^8$-$d^9L$', r'$d^8$-$O$-$d^9$', r'$d^9L$_$d^9L$',
          r'$d^9$-$O$-$d^9L$')
line_styles = ('-g', 'greenyellow', 'navy', 'y', 'b')
marker_styles = ('d', 'h', '^', '^', 's')
for i in range(type_num):
    if i != 4:
        ax.plot(tpds, type_weight[i], line_styles[i], label=labels[i], marker=marker_styles[i], markersize=10)
    else:
        ax.plot(tpds, type_weight[i], line_styles[i], label=labels[i], marker=marker_styles[i], markersize=6)
ax.plot(dft_tpd, 0.003, '*', color='r', markersize=25)
ax.set_xlabel(r'$t_{pd}$', fontsize=30)
ax.set_ylabel('Weight', fontsize=30)
ax.set_xlim([0.78, 1.74])
ax.set_ylim([0, 1.4])
ax.set_xticks(np.arange(0.8, 1.61, 0.2))
ax.set_yticks(np.arange(0, 0.81, 0.2))
ax.tick_params(axis='both', labelsize=24)
ax.legend(loc='best', bbox_to_anchor=(0.50, 0.25, 0.5, 0.5), fontsize=16, framealpha=0.5, edgecolor='black')
fig.tight_layout()

# 子图
ax1 = fig.add_axes((0.24, 0.68, 0.31, 0.28))
ax.text(0.67, 1.3, '(b)', fontsize=30)
upup_dndn = []
updn_updn = []
NiupNiup_NidnOdn = []
with open('max_orb_weight', 'r') as f:
    for line in f:
        if '1th total weight' in line:
            weight = line.split('=')[-1].strip()
            weight = float(weight)
            upup_dndn.append(weight)
        if '2th total weight' in line:
            weight = line.split('=')[-1].strip()
            weight = float(weight)
            updn_updn.append(weight)
        if '3th total weight' in line:
            weight = line.split('=')[-1].strip()
            weight = float(weight)
            NiupNiup_NidnOdn.append(weight)

# 检查读取数据是否正确
# print('upup_dndn')
# for weight in upup_dndn:
#     print(weight)
# print('updn_updn')
# for weight in updn_updn:
#     print(weight)

label_style = ("{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}-{$d_{z^2}^{\\downarrow}d_{x^2}^{\\downarrow}$}(B)",
               "{$d_{z^2}^{\\uparrow}d_{x^2}^{\\downarrow}$}-{$d_{z^2}^{\\uparrow}d_{x^2}^{\\downarrow}$}",
               "{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}-{$d_{z^2}^{\\downarrow}L^{\\downarrow}$(B)}")
ax1.plot(tpds, upup_dndn, color='cyan', marker='D', label=label_style[0], markersize=6)
ax1.plot(tpds, updn_updn, color='pink', marker='o', label=label_style[1], markersize=6)
ax1.plot(tpds, NiupNiup_NidnOdn, color='purple', marker='*', label=label_style[2], markersize=6)

ax1.legend(loc='best', bbox_to_anchor=(0.98, 0.95, 0.1, 0.1), fontsize=16, framealpha=0.5, edgecolor='black')
ax1.plot(dft_tpd, 0.003, '*', color='r', markersize=20)
ax1.set_ylim(0, 0.8)
ax1.set_xticks([0.8, 1.2, 1.6])
ax1.tick_params(labelsize=20)
fig.show()
fig.savefig('fig3b.pdf')
