import math
import numpy as np
import change_pressure as cp

M_PI = math.pi
pressure = cp.pressure
Mc = 2
# Note that Ni-d and O-p orbitals use hole language
# while Nd orbs use electron language
if pressure == 0:
    edCu = {'d3z2r2': 0.046, \
            'dx2y2': 0, \
            'dxy': 0.823, \
            'dxz': 0.706, \
            'dyz': 0.706}
    epNis = np.arange(2.47, 2.471, 1.0)
    epbilayers = np.arange(2.94, 2.941, 1.0)
elif pressure == 4:
    edCu = {'d3z2r2': 0.054, \
            'dx2y2': 0, \
            'dxy': 0.879, \
            'dxz': 0.761, \
            'dyz': 0.761}
    epNis = np.arange(2.56, 2.561, 1.0)
    epbilayers = np.arange(3.03, 3.031, 1.0)
elif pressure == 8:
    edCu = {'d3z2r2': 0.060, \
            'dx2y2': 0, \
            'dxy': 0.920, \
            'dxz': 0.804, \
            'dyz': 0.804}
    epNis = np.arange(2.62, 2.621, 1.0)
    epbilayers = np.arange(3.02, 3.021, 1.0)
elif pressure == 16:
    edCu = {'d3z2r2': 0.072,
            'dx2y2': 0,
            'dxy': 0.997,
            'dxz': 0.887,
            'dyz': 0.887}
    epNis = np.arange(2.75, 2.751, 1.0)
    epbilayers = np.arange(3.14, 3.141, 1.0)
elif pressure == 29.5:
    edCu = {'d3z2r2': 0.095,
            'dx2y2': 0,
            'dxy': 1.06,
            'dxz': 0.94,
            'dyz': 0.94}
    epNis = np.arange(2.9, 2.91, 1.0)
    epbilayers = np.arange(3.24, 3.241, 1.0)

edNi = edCu
epCus = epNis
ANis = np.arange(6.0, 6.01, 1.0)
ACus = ANis
B = 0.15
C = 0.58
# As = np.arange(100, 100.1, 1.0)
# ANis = np.arange(0.0, 0.01, 1.0)
# ACus = np.arange(0.0, 0.01, 1.0)
# B = 0
# C = 0

# Note: tpd and tpp are only amplitude signs are considered separately in hamiltonian.py
# Slater Koster integrals and the overlaps between px and d_x^2-y^2 is sqrt(3) bigger than between px and d_3z^2-r^2
# These two are proportional to tpd,sigma, whereas the hopping between py and d_xy is proportional to tpd,pi

# IMPORTANT: keep all hoppings below positive to avoid confusion
#            hopping signs are considered in dispersion separately
Norb = 5
if Norb == 8 or Norb == 5:
    # tpds = [0.00001]  # for check_CuO4_eigenvalues.py
    if pressure == 0:
        tpds = np.linspace(1.38, 1.38, num=1, endpoint=True)
        # tpds = [3.0]

        tpps = [0.537]
        tapzds = np.linspace(1.48, 1.48, num=1, endpoint=True)
        # tapzds = [2.4]

        tapzps = np.linspace(0.445, 0.445, num=1, endpoint=True)
    if pressure == 4:
        tpds = np.linspace(1.43, 1.43, num=1, endpoint=True)  # [0.25]
        tpps = [0.548]
        tapzds = np.linspace(1.53, 1.53, num=1, endpoint=True)
        tapzps = np.linspace(0.458, 0.458, num=1, endpoint=True)
    if pressure == 8:
        tpds = np.linspace(1.46, 1.46, num=1, endpoint=True)  # [0.25]
        tpps = [0.554]
        tapzds = np.linspace(1.55, 1.55, num=1, endpoint=True)
        tapzps = np.linspace(0.468, 0.468, num=1, endpoint=True)
    if pressure == 16:
        tpds = np.linspace(1.52, 1.52, num=1, endpoint=True)  # [0.25]
        tpps = [0.566]
        tapzds = np.linspace(1.61, 1.61, num=1, endpoint=True)
        tapzps = np.linspace(0.484, 0.484, num=1, endpoint=True)
    if pressure == 29.5:
        tpds = np.linspace(1.58, 1.58, num=1, endpoint=True)
        # tpds = [3.6]

        tpps = [0.562]
        tapzds = np.linspace(1.66, 1.66, num=1, endpoint=True)
        # tapzds = [3.6]

        tapzps = np.linspace(0.487, 0.487, num=1, endpoint=True)
    tz_a1a1 = 0.028
    tz_b1b1 = 0.047

elif Norb == 10 or Norb == 12:
    # pdp = sqrt(3)/4*pds so that tpd(b2)=tpd(b1)/2: see Eskes's thesis and 1990 paper
    # the values of pds and pdp between papers have factor of 2 difference
    # here use Eskes's thesis Page 4
    # also note that tpd ~ pds*sqrt(3)/2
    vals = np.linspace(1.3, 1.3, num=1, endpoint=True)
    pdss = np.asarray(vals) * 2. / np.sqrt(3)
    pdps = np.asarray(pdss) * np.sqrt(3) / 4.
    #     pdss = [0.01]
    #     pdps = [0.01]
    # ------------------------------------------------------------------------------
    # note that tpp ~ (pps+ppp)/2
    # because 3 or 7 orbital bandwidth is 8*tpp while 9 orbital has 4*(pps+ppp)
    pps = 0.9
    ppp = 0.2

    tz_a1a1 = 0.044
    tz_b1b1 = 0.047

#     pps = 0.01
#     ppp = 0.01


if_tz_exist = 2
# if if_tz_exist = 0,tz exist in all orbits.
# if if_tz_exist = 1,tz exist in d orbits.
# if if_tz_exist = 2,tz exist in d3z2r2 orbits.

wmin = -10;
wmax = 30
eta = 0.1
Lanczos_maxiter = 600

# restriction on variational space
reduce_VS = 0
all_A_d8910 = 'd9'
if_H0_rotate_byU = 1
basis_change_type = 'd_double'  # 'all_states' or 'd_double'
if_print_VS_after_basis_change = 0

if_compute_Aw = 0
if if_compute_Aw == 1:
    if_find_lowpeak = 0
    if if_find_lowpeak == 1:
        peak_mode = 'lowest_peak'  # 'lowest_peak' or 'highest_peak' or 'lowest_peak_intensity'
        if_write_lowpeak_ep_tpd = 1
    if_write_Aw = 0
    if_savefig_Aw = 1

if_get_ground_state = 1
if if_get_ground_state == 1:
    # see issue https://github.com/scipy/scipy/issues/5612
    Neval = 30
if_compute_Aw_dd_total = 0

if Norb == 8 or Norb == 10 or Norb == 12:
    Ni_Cu_orbs = ['dx2y2', 'dxy', 'dxz', 'dyz', 'd3z2r2']
elif Norb == 5:
    Ni_Cu_orbs = ['dx2y2', 'd3z2r2']

if Norb == 8 or Norb == 5:
    O1_orbs = ['px']
    O2_orbs = ['py']
    Obilayer_orbs = ['apz']
elif Norb == 10:
    O1_orbs = ['px1', 'py1']
    O2_orbs = ['px2', 'py2']
    Obilayer_orbs = ['apz']
elif Norb == 12:
    O1_orbs = ['px1', 'py1', 'pz1']
    O2_orbs = ['px2', 'py2', 'pz2']
    Obilayer_orbs = ['apz']
O_orbs = O1_orbs + O2_orbs
# sort the list to facilliate the setup of interaction matrix elements
Ni_Cu_orbs.sort()
O1_orbs.sort()
O2_orbs.sort()
O_orbs.sort()
Obilayer_orbs.sort()
print("Ni_Cu_orbs = ", Ni_Cu_orbs)
print("O1_orbs = ", O1_orbs)
print("O2_orbs = ", O2_orbs)
print("Obilayer_orbs = ", Obilayer_orbs)
orbs = Ni_Cu_orbs + O_orbs + Obilayer_orbs
# assert(len(orbs)==Norb)

Upps = [4.0]
Usss = [4.0]
symmetries = ['1A1', '3B1', '3B1', '1A2', '3A2', '1E', '3E']
print("compute A(w) for symmetries = ", symmetries)

