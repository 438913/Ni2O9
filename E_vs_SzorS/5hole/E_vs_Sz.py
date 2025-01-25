import matplotlib.pyplot as plt
import numpy as np

# 基态能量(Sz=1/2, 3/2, 5/2)随着tpd变化的曲线
fig, ax = plt.subplots(figsize=(7.8, 8))
tpd = np.linspace(0.79, 1.738, num=11, endpoint=True)
E_Sz1 = np.genfromtxt('Sz=0.5', dtype=float)
E_Sz2 = np.genfromtxt('Sz=1.5', dtype=float)
E_Sz3 = np.genfromtxt('Sz=2.5', dtype=float)

label_style = ('{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}-$d_{z^2}^{\\uparrow}[d_{x^2}^{\\uparrow}L^{\\uparrow}$](AB)($S=5/2$)',
               '{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}-$d_{z^2}^{\\uparrow}$[$d_{x^2}^{\\uparrow}L^{\\downarrow}$](AB)($S=3/2$)',
               '$d_{z^2}^{\\downarrow}$[$d_{x^2}^{\\downarrow}L^{\\uparrow}$]-{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}(AB)($S=1/2$)')
ax.plot(tpd, E_Sz3 - E_Sz1, color='y', marker='o', markersize=10, label=label_style[0])
ax.plot(tpd, E_Sz2 - E_Sz1, color='r', marker='>', markersize=10, label=label_style[1])
ax.plot([0.79, 1.738], [0.02, 0.02], 'k', linewidth=2, label=label_style[2])

ax.plot(1.58, 0.02, '*', color='r', markersize=25)

# ax.arrow(1.59, 0, 0, 0.15, head_width=0.015, head_length=0.05, fc='k', ec='k')
# ax.text(1.62,0.08, 'J', fontsize=25)

ax.legend(loc='best', fontsize=15, framealpha=0.5, edgecolor='k')
ax.tick_params(axis='both', labelsize=30)
ax.set_xlabel(r'$t_{pd}$', fontsize=40)
ax.set_ylabel(r'$\Delta$E(eV)', fontsize=40)
ax.set_xlim([0.79, 1.738])
ax.set_ylim([0, 5.2])
fig.tight_layout()

ax1 = fig.add_axes((0.5, 0.3, 0.4, 0.3))
ax1.plot(tpd, E_Sz2 - E_Sz1, color='r', marker='>', markersize=10)

ax1.plot([0.8, 1.738], [0.001, 0.001], 'k', linewidth=2)
ax1.plot(1.58, 0.005, '*', color='r', markersize=25)
ax1.arrow(1.58, 0, 0, 0.2, head_width=0.015, head_length=0.02, fc='k', ec='k')
ax1.text(1.62,0.11, 'J', fontsize=25)
ax1.set_xlim([0.8, 1.738])
ax1.set_ylim([0, 0.3])
ax1.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
ax1.set_yticks([0, 0.1, 0.2, 0.3])
ax1.tick_params(labelsize=20)
fig.show()
fig.savefig('fig6.pdf')
