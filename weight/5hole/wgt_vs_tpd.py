import matplotlib.pyplot as plt
import numpy as np

# 用type_weight存储态的类型(比如d8_d8_L)和对应的权重,
# 态的类型作为key, 权重列表作为value
dL_type_list = ('d8_d8L', 'd7_d8', 'd8L_d9L', 'd8_d9L2', 'd8_O1_d9L',
                'd7_d9L', 'd8_O1_d8')
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
            if dL_type in dL_type_list:  # 存储在type_weight中
                idx = dL_type_list.index(dL_type)
                type_weight[idx][i - 1] = weight

# 检验读取数据是否正确
for i in range(type_num):
    print(dL_type_list[i])
    for j in range(tpd_num):
        print(type_weight[i][j])

# 主图
fig, ax = plt.subplots(figsize=(7.8, 8))
labels = (r'$d^8$-$d^8L$', r'$d^7$-$d^8$', r'$d^8L$-$d^9L$', r'$d^8$-$d^9L^2$',
          r'$d^8$-$O$-$d^9L$', r'$d^7$-$d^9L$', r'$d^8$-$O$-$d^8$')
line_styles = ['g', 'peru', 'y', 'r', 'deeppink', 'purple', 'cyan']
marker_styles = ['d', '^', '>', '+', '^', 'x', '*']

for i in range(type_num):
    ax.plot(tpds, type_weight[i], line_styles[i], label=labels[i], marker=marker_styles[i], markersize=10)

ax.plot(dft_tpd, 0.005, '*', color='r', markersize=25)
ax.set_xlabel(r'$t_{pd}$', fontsize=30)
ax.set_ylabel('Weight', fontsize=30)
ax.set_xlim([0.79, 1.738])
ax.set_ylim([0, 1.4])
ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.tick_params(axis='both', labelsize=24)
ax.legend(loc='best', bbox_to_anchor=(0.50, 0.23, 0.5, 0.5), fontsize=16, framealpha=0.5, edgecolor='black')
ax.text(0.67, 1.3, '(b)', fontsize=30)      # 在图的左上角添加(b)
fig.tight_layout()

# 子图
ax1 = fig.add_axes((0.24, 0.58, 0.25, 0.37))
# 读取数据
upup_dndnup = []        # 表示上层Ni的dz2轨道和dx2轨道的空穴自旋均向上，
# 下层Ni的dz2轨道和dx2轨道的空穴自旋均向下，另外的一个O轨道自旋向上
upup_updndn = []
updn_dndnup = []
with open('max_orb_weight', 'r') as f:
    for line in f:
        if '1th total weight' in line:
            weight = line.split('=')[-1].strip()
            weight = float(weight)
            upup_dndnup.append(weight)
        if '2th total weight' in line:
            weight = line.split('=')[-1].strip()
            weight = float(weight)
            upup_updndn.append(weight)
        if '3th total weight' in line:
            weight = line.split('=')[-1].strip()
            weight = float(weight)
            updn_dndnup.append(weight)

label_style = ("{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}-$d_{z^2}^{\\downarrow}$[$d_{x^2}^{\\downarrow}L^{\\uparrow}$](AB)",
              "{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}-{$d_{z^2}^{\\uparrow}d_{x^2}^{\\downarrow}$}$L^{\\downarrow}$(AB)",
               "{$d_{z^2}^{\\uparrow}d_{x^2}^{\\downarrow}$}-$d_{z^2}^{\\downarrow}$[$d_{x^2}^{\\downarrow}L^{\\uparrow}$](AB)")
ax1.plot(tpds, upup_dndnup, 'pink', marker='>', label=label_style[0], markersize=6)
ax1.plot(tpds, upup_updndn, 'red', marker='*', label=label_style[1], markersize=6)
ax1.plot(tpds, updn_dndnup, 'olive', marker='^', label=label_style[2], markersize=6)

ax1.legend(loc='best', bbox_to_anchor=(0.98, 0.9, 0.1, 0.1), fontsize=16, framealpha=0.5, edgecolor='black')
ax1.plot(dft_tpd, 0.003, '*', color='r', markersize=20)
ax1.set_ylim(0, 0.3)
ax1.set_xticks([0.8, 1.2, 1.6])
ax1.tick_params(labelsize=20)
fig.show()
fig.savefig('fig5b.pdf')
