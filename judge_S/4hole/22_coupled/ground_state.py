import subprocess
import os
import sys
import time
import shutil
import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import parameters as pam
import hamiltonian_d9 as ham
import lattice as lat
import variational_space as vs
import utility as util
import lanczos


def reorder_z(slabel):
    '''
    reorder orbs such that d orb is always before p orb and Ni layer (z=1) before Cu layer (z=0)
    '''
    s1 = slabel[0];
    orb1 = slabel[1];
    x1 = slabel[2];
    y1 = slabel[3];
    z1 = slabel[4];
    s2 = slabel[5];
    orb2 = slabel[6];
    x2 = slabel[7];
    y2 = slabel[8];
    z2 = slabel[9];

    state_label = [s1, orb1, x1, y1, z1, s2, orb2, x2, y2, z2]

    if orb1 in pam.Ni_Cu_orbs and orb2 in pam.Ni_Cu_orbs:
        if z2 > z1:
            state_label = [s2, orb2, x2, y2, z2, s1, orb1, x1, y1, z1]
        elif z2 == z1 and orb1 == 'dx2y2' and orb2 == 'd3z2r2':
            state_label = [s2, orb2, x2, y2, z2, s1, orb1, x1, y1, z1]
    elif orb1 in pam.O_orbs and orb2 in pam.Obilayer_orbs:
        state_label = [s2, orb2, x2, y2, z2, s1, orb1, x1, y1, z1]
    elif orb1 in pam.Obilayer_orbs and orb2 in pam.Ni_Cu_orbs:
        state_label = [s2, orb2, x2, y2, z2, s1, orb1, x1, y1, z1]
    elif orb1 in pam.O_orbs and orb2 in pam.Ni_Cu_orbs:
        state_label = [s2, orb2, x2, y2, z2, s1, orb1, x1, y1, z1]

    elif orb1 in pam.O_orbs and orb2 in pam.O_orbs:
        if z2 > z1:
            state_label = [s2, orb2, x2, y2, z2, s1, orb1, x1, y1, z1]

    return state_label


def make_z_canonical(slabel):
    s1 = slabel[0];
    orb1 = slabel[1];
    x1 = slabel[2];
    y1 = slabel[3];
    z1 = slabel[4];
    s2 = slabel[5];
    orb2 = slabel[6];
    x2 = slabel[7];
    y2 = slabel[8];
    z2 = slabel[9];
    s3 = slabel[10];
    orb3 = slabel[11];
    x3 = slabel[12];
    y3 = slabel[13];
    z3 = slabel[14];
    s4 = slabel[15];
    orb4 = slabel[16];
    x4 = slabel[17];
    y4 = slabel[18];
    z4 = slabel[19];
    '''
    For three holes, the original candidate state is c_1*c_2*c_3|vac>
    To generate the canonical_state:
    1. reorder c_1*c_2 if needed to have a tmp12;
    2. reorder tmp12's 2nd hole part and c_3 to have a tmp23;
    3. reorder tmp12's 1st hole part and tmp23's 1st hole part
    '''
    tlabel = [s1, orb1, x1, y1, z1, s2, orb2, x2, y2, z2]
    tmp12 = reorder_z(tlabel)

    tlabel = tmp12[5:10] + [s3, orb3, x3, y3, z3]
    tmp23 = reorder_z(tlabel)

    tlabel = tmp12[0:5] + tmp23[0:5]
    tmp = reorder_z(tlabel)

    slabel = tmp + tmp23[5:10]
    tlabel = slabel[10:15] + [s4, orb4, x4, y4, z4]
    tmp34 = reorder_z(tlabel)

    if tmp34 == tlabel:
        slabel2 = slabel + [s4, orb4, x4, y4, z4]
    elif tmp34 != tlabel:
        tlabel = slabel[5:10] + [s4, orb4, x4, y4, z4]
        tmp24 = reorder_z(tlabel)
        if tmp24 == tlabel:
            slabel2 = slabel[0:10] + [s4, orb4, x4, y4, z4] + slabel[10:15]
        elif tmp24 != tlabel:
            tlabel = slabel[0:5] + [s4, orb4, x4, y4, z4]
            tmp14 = reorder_z(tlabel)
            if tmp14 == tlabel:
                slabel2 = slabel[0:5] + [s4, orb4, x4, y4, z4] + slabel[5:15]
            elif tmp14 != tlabel:
                slabel2 = [s4, orb4, x4, y4, z4] + slabel[0:15]

    return slabel2


def state_classification(state_param):
    """
    Classify the state into different types
    :param state_param: state_param = [(x1, y1, z1, orb1, s1), (x2, y2, z2, orb2, s2)...]
    :return: state_type: string
    """
    Ni_bottom = 0
    Ni_top = 0
    O_bilayer = 0
    O_bottom = 0
    O_top = 0
    dL_type = ['empty', 'empty', 'empty']
    orb_type = ['empty', 'empty', 'empty']
    simple_orb = {'d3z2r2': 'dz2', 'dx2y2': 'dx2', 'dxy': 'dxy', 'dxz': 'dxz', 'dyz': 'dyz'}
    Sz = 0
    # 按照轨道的字典序对state_param排序
    state_param.sort(key=lambda hole: hole[-2])
    for x, y, z, orb, s in state_param:
        if s == 'up':
            Sz += 1 / 2
        else:
            Sz += -1 / 2
        if z == 2:
            if orb in pam.Ni_Cu_orbs:
                Ni_top += 1
                if orb_type[0] == 'empty':
                    orb_type[0] = simple_orb[orb]
                else:
                    orb_type[0] += simple_orb[orb]
            elif orb in pam.O_orbs:
                O_top += 1
        elif z == 1:
            O_bilayer += 1
            if orb_type[1] == 'empty':
                orb_type[1] = orb
            else:
                orb_type[1] += orb
        else:
            if orb in pam.Ni_Cu_orbs:
                Ni_bottom += 1
                if orb_type[2] == 'empty':
                    orb_type[2] = simple_orb[orb]
                else:
                    orb_type[2] += simple_orb[orb]
            elif orb in pam.O_orbs:
                O_bottom += 1
    if Ni_top != 0:
        dL_type[0] = f'd{10 - Ni_top}'
        if O_top != 0:
            dL_type[0] += 'L' if O_top == 1 else f'L{O_top}'
    else:
        if O_top != 0:
            dL_type[0] = 'L' if O_top == 1 else f'L{O_top}'
    if O_bilayer != 0:
        dL_type[1] = 'O' if O_bilayer == 1 else f'O{O_bilayer}'
    if Ni_bottom != 0:
        dL_type[2] = f'd{10 - Ni_bottom}'
        if O_bottom != 0:
            dL_type[2] += 'L' if O_bottom == 1 else f'L{O_bottom}'
    else:
        if O_bottom != 0:
            dL_type[2] = 'L' if O_bottom == 1 else f'L{O_bottom}'
    dL_type = [x for x in dL_type if x != 'empty']
    if len(dL_type) == 3:
        if dL_type[0] > dL_type[-1]:
            dL_type[0], dL_type[-1] = dL_type[-1], dL_type[0]
    else:
        dL_type.sort()
    dL_type = '_'.join(dL_type)
    orb_type = [x for x in orb_type if x != 'empty']
    if len(orb_type) == 3:
        if orb_type[0] > orb_type[-1]:
            orb_type[0], orb_type[-1] = orb_type[-1], orb_type[0]
    else:
        orb_type.sort()
    orb_type = '_'.join(orb_type)
    if Sz == 0:
        Sz = '0'
    elif Sz == 1 or Sz == -1:
        Sz = '+-1'
    else:
        Sz = '+-2'
    if orb_type == '':
        orb_type = f'Sz={Sz}'
    else:
        orb_type = f'{orb_type},Sz={Sz}'
    return dL_type, orb_type


def get_ground_state(matrix, VS, S_Ni_val, Sz_Ni_val, S_Cu_val, Sz_Cu_val, coupled_idx, jm_list, bonding_val):
    '''
    Obtain the ground state info, namely the lowest peak in Aw_dd's component
    in particular how much weight of various d8 channels: a1^2, b1^2, b2^2, e^2
    '''
    t1 = time.time()
    print('start getting ground state')
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')
    vals.sort()
    print('lowest eigenvalue of H from np.linalg.eigsh = ')
    print(vals)
    with open('data/energy_spectrum', 'a') as f:
        f.write('lowest eigenvalue of H from np.linalg.eigsh = \n')
        f.write(str(vals) + '\n\n')
    with open('data/vals', 'a') as f:
        f.write(f'{vals[0]}\n')

    # 计算每个不同的本征值第一次出现的索引，并存储在degen_idx中
    # 对应的简并度即为degen_idx[i+1] - degen_idx[i]
    val_num = 5
    degen_idx = [0]
    for _ in range(val_num):
        for idx in range(degen_idx[-1]+1, pam.Neval):
            if abs(vals[idx] - vals[degen_idx[-1]]) > 1e-4:
                degen_idx.append(idx)
                break

    text_dL_weight = open('data/dL_weight', 'a')
    text_orb_max_weight = open('data/orb_max_weight', 'a')
    for i in range(val_num):
        print(f'Degeneracy of {i}th state is {degen_idx[i + 1]-degen_idx[i]}')
        print('val = ', vals[degen_idx[i]])
        text_dL_weight.write(f'\nDegeneracy of {i}th state is {degen_idx[i + 1] - degen_idx[i]}\n')
        text_orb_max_weight.write(f'\nDegeneracy of {i}th state is {degen_idx[i + 1] - degen_idx[i]}\n')

        weight_average = np.average(abs(vecs[:, degen_idx[i]:degen_idx[i+1]]) ** 2, axis=1)
        ilead = np.argsort(-weight_average)

        total = 0
        # dL_weight用来记录, 例如d8_d9L总的weight,
        # dL_orb用来记录, 例如d8_d9L具体的d轨道和自旋信息
        # dL_orb_weight用来记录, 例如d8_d9L具体的d轨道自旋所对应总的weight
        # dL_orb_istate用来记录, 例如d8_d9L具体的d轨道自旋对应态的索引
        # weight_average是各个态的weight
        dL_weight = {}
        dL_orb = {}
        dL_orb_weight = {}
        dL_orb_istate = {}
        for istate in ilead:
            # state is original state but its orbital info remains after basis change
            weight = weight_average[istate]
            total += weight
            state = VS.get_state(VS.lookup_tbl[istate])
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            s3 = state['hole3_spin']
            s4 = state['hole4_spin']
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            orb3 = state['hole3_orb']
            orb4 = state['hole4_orb']
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']
            x3, y3, z3 = state['hole3_coord']
            x4, y4, z4 = state['hole4_coord']
            bonding = bonding_val[istate]

            input_state = [(x1, y1, z1, orb1, s1), (x2, y2, z2, orb2, s2), (x3, y3, z3, orb3, s3),
                           (x4, y4, z4, orb4, s4)]
            position = f'({x1}, {y1}, {z1}), ({x2}, {y2}, {z2}), ({x3}, {y3}, {z3}), ({x4}, {y4}, {z4})'
            dL_type, orb_type = state_classification(input_state)

            if pam.if_coupled:
                if istate in coupled_idx:
                    orb_type = orb_type.split(',Sz')[0]
                    idx = coupled_idx.index(istate)
                    j, _ = jm_list[idx]
                    orb_type += f',j = {j}'

            # dL_type += f'({bonding})'
            orb_type = position + ',' + orb_type
            dL_orb_type = f'{dL_type},{orb_type}'
            if dL_type in dL_weight:
                dL_weight[dL_type] += weight
            else:
                dL_weight[dL_type] = weight
            if dL_type in dL_orb:
                if orb_type not in dL_orb[dL_type]:
                    dL_orb[dL_type] += [orb_type]
            else:
                dL_orb[dL_type] = [orb_type]
            if dL_orb_type in dL_orb_weight:
                dL_orb_weight[dL_orb_type] += weight
                dL_orb_istate[dL_orb_type] += [istate]
            else:
                dL_orb_weight[dL_orb_type] = weight
                dL_orb_istate[dL_orb_type] = [istate]

        sorted_dL = sorted(dL_weight, key=dL_weight.get, reverse=True)
        for i1, dL_type in enumerate(sorted_dL):
            weight = dL_weight[dL_type]
            if weight < 0.002:
                continue
            if i1 == 0:
                text_orb_max_weight.write(f'{dL_type}: {weight}\n\n')
                print('\n', end='')
            else:
                print('\n\n', end='')
            print(f'{dL_type}: {weight}\n')
            text_dL_weight.write(f'\t{dL_type}: {weight}\n')
            dL_orb_type_list = [f'{dL_type},{orb_type}' for orb_type in dL_orb[dL_type]]
            dL_orb_type_list = sorted(dL_orb_type_list, key=lambda x: dL_orb_weight[x], reverse=True)
            for i2, dL_orb_type in enumerate(dL_orb_type_list):
                weight = dL_orb_weight[dL_orb_type]
                if weight < 1e-3:
                    continue
                orb_type = dL_orb_type.split(',')
                orb_type = ",".join(orb_type[1:])
                if i2 != 0:
                    print('\n', end='')
                print(f'\t{orb_type}: {weight}\n')
                if i1 == 0:
                    text_orb_max_weight.write(f'\t{orb_type}: {weight}\n\n')
                # last_weight = 0
                # weight_num = 1
                for istate in dL_orb_istate[dL_orb_type]:
                    weight = weight_average[istate]
                    state = VS.get_state(VS.lookup_tbl[istate])
                    s1 = state['hole1_spin']
                    s2 = state['hole2_spin']
                    s3 = state['hole3_spin']
                    s4 = state['hole4_spin']
                    orb1 = state['hole1_orb']
                    orb2 = state['hole2_orb']
                    orb3 = state['hole3_orb']
                    orb4 = state['hole4_orb']
                    x1, y1, z1 = state['hole1_coord']
                    x2, y2, z2 = state['hole2_coord']
                    x3, y3, z3 = state['hole3_coord']
                    x4, y4, z4 = state['hole4_coord']
                    output = f'\t({x1}, {y1}, {z1}, {orb1}, {s1}), ({x2}, {y2}, {z2}, {orb2}, {s2}), ' +\
                             f'({x3}, {y3}, {z3}, {orb3}, {s3})\n\t({x4}, {y4}, {z4}, {orb4}, {s4})\n' + \
                             f'\tS_Ni2 = {S_Ni_val[istate]}, Sz_Ni2 = {Sz_Ni_val[istate]}, ' + \
                             f'S_Ni0 = {S_Cu_val[istate]}, Sz_Ni0 = {Sz_Cu_val[istate]}\n' + \
                             f'\tvector = {vecs[istate, degen_idx[i]:degen_idx[i+1]]}, weight = {weight}\n\t'
                    if pam.if_coupled:
                        if istate in coupled_idx:
                            idx = coupled_idx.index(istate)
                            output = (
                                f'\t({x1}, {y1}, {z1}, {orb1}, {s1}), ({x2}, {y2}, {z2}, {orb2}, {s2}), ({x3}, {y3}, {z3}, {orb3}, {s3})\n'
                                f'\t({x4}, {y4}, {z4}, {orb4}, {s4})\n'
                                f'\tj = {jm_list[idx][0]}, m = {jm_list[idx][1]}, bonding = {bonding_val[istate]}, weight = {weight}\n')
                        else:
                            output = f'\t({x1}, {y1}, {z1}, {orb1}, {s1}), ({x2}, {y2}, {z2}, {orb2}, {s2}), ' +\
                                     f'({x3}, {y3}, {z3}, {orb3}, {s3})\n\t({x4}, {y4}, {z4}, {orb4}, {s4})\n' + \
                                     f'\tS_Ni1 = {S_Ni_val[istate]}, Sz_Ni1 = {Sz_Ni_val[istate]}, ' + \
                                     f'S_Ni2 = {S_Cu_val[istate]}, Sz_Ni2 = {Sz_Cu_val[istate]},' + \
                                     f'bonding = {bonding_val[istate]}, weight = {weight}\n'
                    print(output)
                    if i1 == 0:
                        text_orb_max_weight.write(output + '\n')
                # 处理最后一次循环
                # if weight_num > 1:
                #     print(f'\tthe same weight number = {weight_num}\n')
                #     if i1 == 0:
                #         text_orb_max_weight.write(f'\tthe same weight number = {weight_num}\n\n')
    text_dL_weight.write('\n')
    text_dL_weight.close()
    text_orb_max_weight.write('\n\n')
    text_orb_max_weight.close()
