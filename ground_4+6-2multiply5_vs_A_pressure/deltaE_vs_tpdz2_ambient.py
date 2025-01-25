import matplotlib.pyplot as plt
import numpy as np
data_4hole = []
data_5hole = []
data_6hole = []
# 空穴数目
hole_num = [4, 5, 6]
# 按照每行读取基态能量数据, 1-5行为A=5的数据,
# 6-10行为A=6的数据, 11-15行为A=7的数据
title_name = r'$\Delta E$ vs $t_{pdx2}/t_{pdz2}$ under ambient pressure'
for num in hole_num:
    with open(f'./variable_tpdz2/val_{num}hole', 'r') as file:
        for line in file:
            number = line.strip()
            if number is None:
                continue
            number = float(number)
            if num == 4:
                data_4hole.append(number)
            elif num == 5:
                data_5hole.append(number)
            elif num == 6:
                data_6hole.append(number)
# 打印读取的数据，并检验数据是否正确
print(data_4hole)
print(data_5hole)
print(data_6hole)
tpdz2 = np.linspace(0.95, 1.05, num=11, endpoint=True)
E_4holes = {'A=5': data_4hole[:11],
            'A=6': data_4hole[11: 22],
            'A=7': data_4hole[22:]}

E_5holes = {'A=5': data_5hole[:11],
            'A=6': data_5hole[11: 22],
            'A=7': data_5hole[22:]}

E_6holes = {'A=5': data_6hole[:11],
            'A=6': data_6hole[11: 22],
            'A=7': data_6hole[22:]}

# 绘图
fig, ax = plt.subplots()
line_styles = {'A=5': 'b', 'A=6': 'y', 'A=7': 'r'}
Marker_styles = {'A=5': 'o', 'A=6': 's', 'A=7': 'D'}
# 依次绘制A = 4, 5, 6所对应的4个空穴 + 6个空穴 - 2倍5个空穴的能量差
for key in E_4holes.keys():
    deltaE = np.array(E_4holes[key]) + np.array(E_6holes[key]) - 2 * np.array(E_5holes[key])
    # 将单位转换为meV
    # deltaE = 1000 * deltaE
    print(deltaE)
    ax.plot(tpdz2, deltaE, line_styles[key], marker=Marker_styles[key], markersize=6, label=key)
# 画 x = 0的横线
# ax.plot([0, 30], [0, 0], 'k', linewidth=1.0)

ax.set_title(title_name)
# ax.set_xlabel('P(GPa)', fontsize=15)
ax.set_xlabel(r'$t_{pdx2}/t_{pdz2}$', fontsize=15)
# ax.set_ylabel(r'$\Delta$E(meV)', fontsize=15)
ax.set_ylabel(r'$\Delta$E(eV)', fontsize=15)
# ax.set_xlim(0, 30)
# ax.set_ylim(0.15, 0.35)
# ax.set_ylim(0, 30)
# ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
# ax.set_yticks([0.15, 0.2, 0.25, 0.3, 0.35])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.legend(fontsize=16)
fig.tight_layout()
fig.show()
fig.savefig('deltaE_vs_tpdz2_under_ambient_pressure.pdf')
