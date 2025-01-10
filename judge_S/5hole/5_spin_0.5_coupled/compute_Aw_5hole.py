import math
import os
import numpy as np
from scipy.sparse.linalg import inv
#from numpy.linalg import inv
import scipy.sparse as sps
import scipy.sparse.linalg
from scipy import integrate
import sys
import matplotlib.pyplot as plt
sys.path.append('../../src/')
from pylab import *
import shutil

import parameters as pam
import lattice as lat
import variational_space as vs
import hamiltonian as ham
import basis_change as basis
import get_state as getstate
import utility as util
import plotfig as fig
import ground_state as gs
import lanczos
import time
start_time = time.time()
M_PI = math.pi

#####################################
def compute_Aw_main(ANi,ACu,epCu,epNi,epbilayer,tpd,tpp,tapzd,tapzp,tz_a1a1,tz_b1b1,pds,pdp,pps,ppp,Upp,Uss,\
                    d_Ni_double,d_Cu_double,p_double,apz_double,double_Ni_part,hole345_Ni_part, double_Cu_part,\
                    hole345_Cu_part, idx_Ni,idx_Cu, U_Ni, \
                    S_Ni_val, Sz_Ni_val, AorB_Ni_sym, \
                    U_Cu, S_Cu_val, Sz_Cu_val, AorB_Cu_sym):
    if Norb==8 or Norb==5:
        fname = 'epCu'+str(epCu)+'epNi'+str(epNi)+'_tpd'+str(tpd)+'_tpp'+str(tpp) \
                  +'_Mc'+str(Mc)+'_eta'+str(eta) \
                  +'_tz_a1a1' +str(tz_a1a1)  +'_tz_b1b1' +str(tz_b1b1) +'_Upp'+str(Upp)\
                  + '_Uss'+str(Uss) + '_tapzd'+str(tapzd)+ '_tapzp'+str(tapzp)
        flowpeak = 'Norb'+str(Norb)+'_tpp'+str(tpp)+'_Mc'+str(Mc)+'_eta'+str(eta)
    elif Norb==10 or Norb==11 or Norb==12:
        fname = 'epCu'+str(epCu)+'epNi'+str(epNi)+'_pdp'+str(pdp)+'_pps'+str(pps)+'_ppp'+str(ppp) \
                  +'_Mc'+str(Mc)+'_Norb'+str(Norb)+'_eta'+str(eta) \
                  +'_ANi'+str(ANi) + '_ACu'+str(ACu) + '_B'+str(B) + '_C'+str(C)
        flowpeak = 'Norb'+str(Norb)+'_pps'+str(pps)+'_ppp'+str(ppp)+'_Mc'+str(Mc)+'_eta'+str(eta)


    w_vals = np.arange(pam.wmin, pam.wmax, pam.eta)
    Aw = np.zeros(len(w_vals))
    Aw_dd_total = np.zeros(len(w_vals))
    Aw_d8_total = np.zeros(len(w_vals))

    # set up H0
    if Norb==8 or Norb==5:
        tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac, tpp_nn_hop_fac \
                                   = ham.set_tpd_tpp(Norb,tpd,tpp,0,0,0,0)
    elif Norb==10 or Norb==12:
        tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac, tpp_nn_hop_fac \
                                   = ham.set_tpd_tpp(Norb,0,0,pds,pdp,pps,ppp)

    if Norb==5:
        tapzd_nn_hop_dir, tapzd_orbs, tapzd_nn_hop_fac, tapzp_nn_hop_dir, tapzp_orbs, tapzp_nn_hop_fac\
                                   = ham.set_tdO_tpO(Norb,tapzd,tapzp)


    tz_fac = ham.set_tz(Norb,if_tz_exist,tz_a1a1,tz_b1b1)

    T_pd   = ham.create_tpd_nn_matrix(VS,tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac)
    T_pp   = ham.create_tpp_nn_matrix(VS,tpp_nn_hop_fac)
    T_z    = ham.create_tz_matrix(VS,tz_fac)
    T_apzd   = ham.create_tapzd_nn_matrix(VS,tapzd_nn_hop_dir, tapzd_orbs, tapzd_nn_hop_fac)
    T_apzp   = ham.create_tapzp_nn_matrix(VS,tapzp_nn_hop_dir, tapzp_orbs, tapzp_nn_hop_fac)
    Esite  = ham.create_edep_diag_matrix(VS,ANi,ACu, edNi, edCu, epNi,epCu,epbilayer)

    H0 = T_pd + T_pp + T_z + T_apzd + T_apzp + Esite
#     H0 = T_pd
    print("H0 %s seconds ---" % (time.time() - start_time))

    '''
    Below probably not necessary to do the rotation by multiplying U and U_d
    the basis_change.py is only for label the state as singlet or triplet
    and assign the interaction matrix
    '''
    if pam.if_H0_rotate_byU==1:
        H0_Ni_new = U_Ni_d.dot(H0.dot(U_Ni))


    if Norb==5 or Norb==6 or Norb==10 or Norb==11 or Norb==12:
        Hint_Ni = ham.create_interaction_matrix_ALL_syms(VS,d_Ni_double,double_Ni_part, idx_Ni, hole345_Ni_part,  \
                                                      S_Ni_val, Sz_Ni_val,AorB_Ni_sym, ACu, ANi)
        Hint_Cu = ham.create_interaction_matrix_ALL_syms(VS,d_Cu_double,double_Cu_part, idx_Cu, hole345_Cu_part, \
                                                      S_Cu_val, Sz_Cu_val,AorB_Cu_sym, ACu, ANi)
        Hint_po = ham.create_interaction_matrix_po(VS,p_double,apz_double, Upp, Uss)

        if pam.if_H0_rotate_byU==1:
            H_Ni = H0_Ni_new + Hint_Ni

            # continue rotate the basis for setting Cu layer's interaction (d_Cu_double)
            H0_Cu_new = U_Cu_d.dot(H_Ni.dot(U_Cu))
            H = H0_Cu_new + Hint_Cu+ Hint_po
        else:
            H = H0 + Hint_Ni + Hint_Cu+ Hint_po

        print("interaction %s seconds ---" % (time.time() - start_time))
#         if pam.basis_change_type=='all_states':
#             U_other, S_Ni_val, Sz_Ni_val, S_Cu_val, Sz_Cu_val \
#                 = basis.create_singlet_triplet_basis_change_matrix_other_states(VS, d_Ni_double, d_Cu_double)
#             U_other_d = (U_other.conjugate()).transpose()

#             H = U_other_d.dot(H.dot(U_other))

        # 从2个空穴的耦合表象变换到1个空穴的非耦合表象
        H = U_Cu @ H @ U_Cu_d
        H = U_Ni @ H @ U_Ni_d
        # 把这部分((0, 0, 2, 'd3z2r2'), (0, 0, 2, 'dx2y2'), (0, 0, 1, 'apz'), (0, 0, 0, 'd3z2r2'), (0, 0, 0, 'dx2y2'))
        # 变换到5个空穴的耦合表象
        H_coupled =U_coupled_d @ H @ U_coupled

        # 做bonding, anti_bonding变换
        H_bond = U_bond_d.dot(H_coupled.dot(U_bond))


        H_bond.tocsr()

        ####################################################################################
        # compute GS only for turning on full interactions
        if pam.if_get_ground_state==1:
            gs.get_ground_state(H_coupled, VS, S_Ni_val,Sz_Ni_val,S_Cu_val,Sz_Cu_val,coupled_idx, jm_list,bonding_val)
        print("ground state %s seconds ---" % (time.time() - start_time))

#             if Norb==8:
#                 util.write_GS('Egs_'+flowpeak+'.txt',A,ep,tpd,vals[0])
#             elif Norb==10 or Norb==11 or Norb==12:
#                 util.write_GS2('Egs_'+flowpeak+'.txt',A,ep,pds,pdp,vals[0])
        #########################################################################
        '''
        Compute A(w) for various states
        '''
        if pam.if_compute_Aw==1:
            clf()
#             # compute d8
#             fig.compute_Aw_d8_sym(H, VS, d_double, S_val, Sz_val, AorB_sym, A, w_vals, "Aw_d8_sym_", fname)

            #compute d9Ld9L
            a1b1_O_a1b1_state_indices, a1b1_O_a1b1_state_labels, \
                    = getstate.get_d8Od8_state_indices(VS)

            a1b1_a1b1L_state_indices, a1b1_a1b1L_state_labels, \
                    = getstate.get_d8d8L_state_indices(VS)
            fig.compute_Aw1(H, VS, w_vals,  a1b1_a1b1L_state_indices,  a1b1_a1b1L_state_labels,  a1b1_O_a1b1_state_indices,  a1b1_O_a1b1_state_labels,"Aw_5hole_", fname)
#             #compute d9d9L2
#             d9d9L2_a1_b1L2_state_indices, d9d9L2_a1_b1L2_state_labels, \
#                     = getstate.get_d9d9L2_state_indices(VS)
#             fig.compute_Aw1(H, VS, w_vals,  d9d9L2_a1_b1L2_state_indices, d9d9L2_a1_b1L2_state_labels, "Aw_d9d9L2_a1_b1L2__", fname)
#             #compute d9L2d9        
#             d9L2d9_a1L2_b1_state_indices, d9L2d9_a1L2_b1_state_labels, \
#                     = getstate.get_d9L2d9_state_indices(VS)
#             fig.compute_Aw1(H, VS, w_vals,  d9L2d9_a1L2_b1_state_indices, d9L2d9_a1L2_b1_state_labels, "Aw_d9L2d9_a1L2_b1__", fname)

#             fig.compute_Aw_d8d8_sym(H, VS, d_Ni_double, S_Ni_val, Sz_Ni_val, AorB_Ni_sym, ANi, S_Cu_val, \
#                                     Sz_Cu_val,AorB_Cu_sym, ACu, w_vals, "Aw_d8d8_sym_", fname)    
##########################################################################
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    for filename in os.listdir('data'):
        file_path = os.path.join('data', filename)
        with open(file_path, 'w') as f:
            f.truncate(0)
    for ANi in pam.ANis:
        ACu = ANi
        Mc = pam.Mc
        pressures = pam.pressures
        edNis = pam.edNis
        epNis_list = pam.epNis_list
        epbilayers_list = pam.epbilayers_list
        tpds_list = pam.tpds_list
        tpps_list = pam.tpps_list
        tapzds_list = pam.tapzds_list
        tapzps_list = pam.tapzps_list
        for i in [4]:
            print ('Mc=',Mc)
            pressure = pressures[i]

            Norb = pam.Norb
            eta  = pam.eta
            edNi = {key: item[i] for key, item in edNis.items()}
            edCu = edNi

            ANis = pam.ANis
            ACus = pam.ACus
            B  = pam.B
            C  = pam.C

            epNis = [epNis_list[i]]
            epCus = epNis
            epbilayers = [epbilayers_list[i]]
            tpds = [tpds_list[i]]
            # tpds = tpds_list
            tpps = [tpps_list[i]]
            tapzds = [tapzds_list[i]]
            tapzps = [tapzps_list[i]]

            tz_a1a1 = pam.tz_a1a1
            tz_b1b1 = pam.tz_b1b1

            if_tz_exist  = pam.if_tz_exist

            # set up VS
            VS = vs.VariationalSpace(Mc)
        #     basis.count_VS(VS)
            print("vs %s seconds ---" % (time.time() - start_time))

            d_Ni_double, idx_Ni, hole345_Ni_part,  double_Ni_part, \
            d_Cu_double, idx_Cu, hole345_Cu_part,  double_Cu_part, \
            p_double,apz_double = ham.get_double_occu_list(VS)

            # change the basis for d_double states to be singlet/triplet

            if pam.basis_change_type =='all_states':
                U_Ni,S_Ni_val, Sz_Ni_val, AorB_Ni_sym,\
                                                =  basis.create_singlet_triplet_basis_change_matrix \
                                                (VS, double_Ni_part, idx_Ni, hole345_Ni_part,d_Ni_double, d_Cu_double, 'Ni')

                U_Cu,S_Cu_val, Sz_Cu_val, AorB_Cu_sym,\
                                                =  basis.create_singlet_triplet_basis_change_matrix \
                                                (VS, double_Cu_part, idx_Cu, hole345_Cu_part,d_Ni_double, d_Cu_double, 'Cu')

            if pam.basis_change_type =='d_double':
                print("2 %s seconds ---" % (time.time() - start_time))
                U_Ni,S_Ni_val, Sz_Ni_val, AorB_Ni_sym,\
                             =  basis.create_singlet_triplet_basis_change_matrix_d_double(VS, d_Ni_double, double_Ni_part, idx_Ni, hole345_Ni_part)
                print("3 %s seconds ---" % (time.time() - start_time))
                U_Cu,S_Cu_val, Sz_Cu_val, AorB_Cu_sym,\
                             =  basis.create_singlet_triplet_basis_change_matrix_d_double(VS, d_Cu_double, double_Cu_part, idx_Cu, hole345_Cu_part)

                U_coupled, coupled_idx, jm_list = basis.create_coupled_representation_matrix(VS, S_Ni_val, Sz_Ni_val,
                                                                                             S_Cu_val, Sz_Cu_val)


            print("1 %s seconds ---" % (time.time() - start_time))
            U_bond,bonding_val = basis.create_bonding_anti_bonding_basis_change_matrix(VS)
            print("basis %s seconds ---" % (time.time() - start_time))


            if pam.if_print_VS_after_basis_change==1:
                basis.print_VS_after_basis_change(VS,S_val,Sz_val)

        #         U = U_Ni+U_Cu
        #         print(U_Ni)
        #         print(U_Cu)
            U_Ni_d = (U_Ni.conjugate()).transpose()
            U_Cu_d = (U_Cu.conjugate()).transpose()
            U_coupled_d = (U_coupled.conjugate()).transpose()
            U_bond_d = (U_bond.conjugate()).transpose()

            # check if U if unitary
        #     util.checkU_unitary(U_Ni,U_Ni_d)

            if Norb==8 or Norb==5:
                    for tpd in tpds:
                        for tapzd in tapzds:
                            for tapzp in tapzps:
                                for epCu in epCus:
                                    for epNi in epNis:
                                        for epbilayer in epbilayers:
                #                        util.get_atomic_d8_energy(ANi,B,C)
                                            for tpp in tpps:
                                                for Upp in pam.Upps:
                                                    for Uss in pam.Usss:
                                                        print ('===================================================')
                                                        print ('ANi=',ANi, 'ACu=',ACu,'epCu=', epCu, 'epNi=',epNi,\
                                                               ' tpd=',tpd,' tpp=',tpp,' Upp=',Upp,' Uss=',Uss ,'tz_a1a1=',tz_a1a1,'tz_b1b1=',tz_b1b1,\
                                                               'tapzd=',tapzd,'tapzp=',tapzp)
                                                        file_path_list = ('./data/dL_weight', './data/energy_spectrum', './data/orb_max_weight')
                                                        for file_path in file_path_list:
                                                            with open(file_path, 'a') as f:
                                                                f.write(f'A = {ANi}, B = {B}, C = {C}, ed = {edNi}\n'
                                                                        f'ep = {epCu}, eo = {epbilayer}, tpp = {tpp}, tpo = {tapzp}, '
                                                                        f'tz_a1a1 = {tz_a1a1}, tz_b1b1 = {tz_b1b1}, Upp = {Upp}, Uoo = {Uss}\n'
                                                                        f'tpd = {tpd}, tdo = {tapzd}\n')

                                                        compute_Aw_main(ANi,ACu,epCu,epNi,epbilayer,tpd,tpp,tapzd,tapzp,tz_a1a1,tz_b1b1,0,0,0,0,Upp,Uss,\
                                                                        d_Ni_double,d_Cu_double,p_double,apz_double,double_Ni_part,hole345_Ni_part,\
                                                                        double_Cu_part,hole345_Cu_part, idx_Ni,idx_Cu, \
                                                                        U_Ni, S_Ni_val, Sz_Ni_val, AorB_Ni_sym ,U_Cu, \
                                                                        S_Cu_val, Sz_Cu_val, AorB_Cu_sym)


    print("--- %s seconds ---" % (time.time() - start_time))