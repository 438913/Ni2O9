import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7.8, 8))
tpd = np.linspace(0.79, 1.738, num=11, endpoint=True)
E_Sz0 = np.genfromtxt('Sz=0', dtype=float)
E_Sz1 = np.genfromtxt('Sz=1', dtype=float)
E_Sz2 = np.genfromtxt('Sz=2', dtype=float)
E_Sz3 = np.genfromtxt('Sz=3', dtype=float)

label_style = ('{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}$L^{\\uparrow}$-{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}$L^{\\uparrow}$(A)($S=3$)',
               '{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}$L^{\\uparrow}$-$d_{z^2}^{\\uparrow}$[$d_{x^2}^{\\uparrow}L^{\\downarrow}$](B)($S=2$)',
               '$d_{z^2}^{\\uparrow}$[$d_{x^2}^{\\uparrow}L^{\\downarrow}$]-$d_{z^2}^{\\uparrow}$[$d_{x^2}^{\\uparrow}L^{\\downarrow}$](A)($S=1$)',
               '$d_{z^2}^{\\uparrow}$[$d_{x^2}^{\\uparrow}L^{\\downarrow}$]-$d_{z^2}^{\\downarrow}$[$d_{x^2}^{\\downarrow}L^{\\uparrow}$](B)($S=0$)')
ax.plot(tpd, E_Sz3 - E_Sz0, color='cyan', marker='o', markersize=10, label=label_style[0])
ax.plot(tpd, E_Sz2 - E_Sz0, color='pink', marker='>', markersize=10, label=label_style[1])
ax.plot(tpd, E_Sz1 - E_Sz0, color='brown', marker='>', markersize=10, label=label_style[2])
ax.plot([0.79, 1.738], [0.05, 0.05], 'k', linewidth=2.4, label=label_style[3])

ax.plot(1.58, 0.04, '*', color='r', markersize=25)

ax.legend(loc='best', fontsize='15', framealpha=0.5, edgecolor='k')
ax.tick_params(axis='both', labelsize=30)
ax.set_xlabel(r'$t_{pd}$', fontsize=40)
ax.set_ylabel(r'$\Delta$E(eV)', fontsize=40)
ax.set_xlim([0.79, 1.738])
ax.set_ylim([0, 10.9])
fig.tight_layout()

# 子图部分
ax1 = fig.add_axes((0.65, 0.25, 0.25, 0.17))
ax1.plot(tpd, E_Sz1-E_Sz0, color='brown', marker='>', markersize=10)

ax1.plot([0.79, 1.738], [0.006, 0.006], 'k', linewidth=2.5)
ax1.plot(1.58, 0.01, '*', color='r', markersize=25)
ax1.arrow(1.58, 0, 0, 0.25, head_width=0.015, head_length=0.03, fc='k', ec='k')
ax1.text(1.62, 0.11, 'J', fontsize=25)
ax1.set_xlim([1.2, 1.738])
ax1.set_ylim([0, 0.4])
ax1.set_xticks([0.8, 1.2, 1.6])
ax1.set_yticks([0, 0.2, 0.4])
ax1.tick_params(labelsize=20)
fig.show()
fig.savefig('fig8.pdf')
