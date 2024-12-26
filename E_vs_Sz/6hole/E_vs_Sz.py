import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7.8, 8))
tpd = np.linspace(0, 2.4, num=9, endpoint=True)
E_Sz1 = np.genfromtxt('Sz=0.5_tpd', dtype=float)
E_Sz2 = np.genfromtxt('Sz=1.5_tpd', dtype=float)
E_Sz3 = np.genfromtxt('Sz=2.5_tpd', dtype=float)
ax.plot(tpd, E_Sz2 - E_Sz1, color='y', marker='o', markersize=10)
ax.plot(tpd, E_Sz3 - E_Sz1, color='r', marker='>', markersize=10)
ax.tick_params(axis='both', labelsize=30)
ax.set_xlabel(r'$t_{pd}$', fontsize=40)
ax.set_ylabel(r'$\Delta$E(eV)', fontsize=40)
ax.set_xlim([0, 2.4])
ax.set_ylim([0, 5])
fig.tight_layout()
fig.show()
