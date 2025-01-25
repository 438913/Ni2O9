import matplotlib.pyplot as plt
import numpy as np
import re
data_4hole = []
data_5hole = []
data_6hole = []
# 空穴数目
hole_num = [4, 5, 6]
# 按照每行读取基态能量数据, 1-5行为A=5的数据,
# 6-10行为A=6的数据, 11-15行为A=7的数据
title_name = 'DFT tpdz2'
for num in hole_num:
    with open(f'./dft_tpdz2/val_{num}hole', 'r') as file:
        for line in file:
            number = re.search(r'-?\d+(\.\d+)?', line)
            if number is None:
                continue
            number = float(number.group(0))
            if num == 4:
                data_4hole.append(float(number))
            elif num == 5:
                data_5hole.append(float(number))
            elif num == 6:
                data_6hole.append(float(number))
# 打印读取的数据，并检验数据是否正确
print(data_4hole)
print(data_5hole)
print(data_6hole)
# 压力0, 4, 8, 16, 29.5 GPa
pressure = np.array([0, 4, 8, 16, 29.5])
# 将读取的数据每5行进行分组, 分别代表A=5, A=6, A=7
E_4holes = {'A=5': data_4hole[0: 5],
            'A=6': data_4hole[5: 10],
            'A=7': data_4hole[10: 15]}

E_5holes = {'A=5': data_5hole[0: 5],
            'A=6': data_5hole[5: 10],
            'A=7': data_5hole[10: 15]}

E_6holes = {'A=5': data_6hole[0: 5],
            'A=6': data_6hole[5: 10],
            'A=7': data_6hole[10: 15]}

# 绘图
fig, ax = plt.subplots()
line_styles = {'A=5': 'b', 'A=6': 'y', 'A=7': 'r'}
Marker_styles = {'A=5': 'o', 'A=6': 's', 'A=7': 'D'}
# 依次绘制A = 4, 5, 6所对应的4个空穴 + 6个空穴 - 2倍5个空穴的能量差
for key in E_4holes.keys():
    deltaE = np.array(E_4holes[key]) + np.array(E_6holes[key]) - 2 * np.array(E_5holes[key])
    # 将单位转换为meV
    deltaE = 1000 * deltaE
    print(deltaE)
    ax.plot(pressure, deltaE, line_styles[key], marker=Marker_styles[key], markersize=6, label=key)
# 画 x = 0的横线
ax.plot([0, 30], [0, 0], 'k', linewidth=1.0)

# ax.set_title(title_name)
ax.set_xlabel('P(GPa)', fontsize=15)
ax.set_ylabel(r'$\Delta$E(meV)', fontsize=15)
# ax.set_ylabel(r'$\Delta$E(eV)', fontsize=15)
ax.set_xlim(0, 30)
# ax.set_ylim(0.15, 0.35)
ax.set_ylim(-5, 20)
ax.set_yticks([-5, 0, 5, 10, 15, 20])
# ax.set_yticks([0.15, 0.2, 0.25, 0.3, 0.35])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.legend(fontsize=16)
fig.tight_layout()
fig.show()
fig.savefig('fig10.pdf')

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
for key in E_4holes.keys():
    E1 = np.array(E_4holes[key])
    ax1.plot(pressure, E1, line_styles[key], marker=Marker_styles[key], markersize=6, label=key)
    E2 = np.array(E_5holes[key])
    ax2.plot(pressure, E2, line_styles[key], marker=Marker_styles[key], markersize=6, label=key)
    E3 = np.array(E_6holes[key])
    ax3.plot(pressure, E3, line_styles[key], marker=Marker_styles[key], markersize=6, label=key)

ax1.set_xlabel('P(GPa)', fontsize=15)
ax2.set_xlabel('P(GPa)', fontsize=15)
ax3.set_xlabel('P(GPa)', fontsize=15)
ax1.set_ylabel(r'E(eV)', fontsize=15)
ax2.set_ylabel(r'E(eV)', fontsize=15)
ax3.set_ylabel(r'E(eV)', fontsize=15)

ax1.set_title('4 hole', fontsize=18)
ax2.set_title('5 hole', fontsize=18)
ax3.set_title('6 hole', fontsize=18)

ax1.set_xlim(0, 30)
ax2.set_xlim(0, 30)
ax3.set_xlim(0, 30)
ax1.set_ylim(0, 12)
ax2.set_ylim(0, 12)
ax3.set_ylim(0, 12)
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
ax3.tick_params(labelsize=14)

ax1.legend(fontsize=16)
ax2.legend(fontsize=16)
ax3.legend(fontsize=16)

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()

fig1.show()
fig2.show()
fig3.show()
fig1.savefig('fig10a.pdf')
fig2.savefig('fig10b.pdf')
fig3.savefig('fig10c.pdf')
