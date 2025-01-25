import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7.8, 8))
tpd = np.linspace(0.79, 1.738, num=11, endpoint=True)
E_Sz0 = np.genfromtxt('Sz=0', dtype=float)
E_Sz1 = np.genfromtxt('Sz=1', dtype=float)
E_Sz2 = np.genfromtxt('Sz=2', dtype=float)

label_style = ('{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}-{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}($S=2$)',
               '{$d_{z^2}^{\\uparrow}d_{x^2}^{\\downarrow}$}-{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}(A)($S=1$)',
               '{$d_{z^2}^{\\uparrow}d_{x^2}^{\\uparrow}$}-{$d_{z^2}^{\\downarrow}d_{x^2}^{\\downarrow}$}(B)($S=0$)')

ax.plot(tpd, E_Sz2 - E_Sz0, color='y', marker='>', markersize=10, label=label_style[0])
ax.plot(tpd, E_Sz1 - E_Sz0, color='b', marker='o', markersize=10, label=label_style[1])
ax.plot([0.79, 1.738], [0.001, 0.001], 'k', linewidth=2.5, label=label_style[2])

ax.arrow(1.58, 0, 0, 0.044, head_width=0.02, head_length=0.005, fc='k', ec='k')
ax.text(1.62, 0.02, 'J', fontsize=30)
ax.plot(1.58, 0.001, '*', color='r', markersize=25)

ax.tick_params(axis='both', labelsize=30)
ax.set_xlabel(r'$t_{pd}$', fontsize=40)
ax.set_ylabel(r'$\Delta$E(eV)', fontsize=40)
ax.set_xlim([0.79, 1.74])
ax.set_ylim([0, 0.20])
ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
ax.set_yticks([0, 0.04, 0.08, 0.12, 0.16, 0.20])
ax.legend(loc='best', fontsize=16, framealpha=0.5, edgecolor='black')
fig.tight_layout()
fig.show()
fig.savefig('fig4.pdf')
