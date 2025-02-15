from itertools import combinations
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg


def create_lookup_tbl():
    """
    创建lookup_tbl，用于查找态所对应的索引
    :return:
    """
    hole_list = [(0, 'd3z2r2'), (0, 'dx2y2'), (1, 'apz'), (2, 'd3z2r2'), (2, 'dx2y2')]
    hole_s_list = []
    for hole in hole_list:
        for s in ['up', 'dn']:
            hole_s = hole + (s,)
            hole_s_list.append(hole_s)
    hole_list = sorted(hole_s_list)
    lookup_tbl = combinations(hole_list, hole_num)
    # filter_lookup_tbl = []
    # for state in lookup_tbl:
    #     Ni0_num = 0
    #     Ni2_num = 0
    #     for hole in state:
    #         z = hole[0]
    #         if z == 0:
    #             Ni0_num += 1
    #         elif z == 2:
    #             Ni2_num +=  1
    #     if Ni0_num < 4 and Ni2_num < 4:
    #         filter_lookup_tbl.append(state)
    #     else:
    #         print(state)
    # filter_lookup_tbl = tuple(filter_lookup_tbl)
    return tuple(lookup_tbl)


def count_inversion(state):
    """
    计算元组的逆序数(即经过多少次交换使得元组为升序)
    :param state: state = (hole1, hole2, ....)
    :return: 元组的逆序数 inversion
    """
    inversion = 0
    for i in range(1, hole_num):
        behind_hole = state[i]
        for front_hole in state[:i]:
            if front_hole > behind_hole:
                inversion += 1
    return inversion


def get_interaction_mat(A, B, C, sym):
    """

    :param A:
    :param B:
    :param C:
    :param sym:
    :return:
    """
    if sym == '1A1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 1
        state_order = {('d3z2r2', 'd3z2r2'): 0,
                       ('dx2y2', 'dx2y2'): 1}
        interaction_mat = [[A + 4. * B + 3. * C, 4. * B + C],
                           [4. * B + C, A + 4. * B + 3. * C]]
    if sym == '1B1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = -1
        state_order = {('d3z2r2', 'dx2y2'): 0}
        interaction_mat = [[A + 2. * C]]
    if sym == '3B1':
        Stot = 1
        Sz_set = [-1, 0, 1]
        AorB_sym = 0
        state_order = {('d3z2r2', 'dx2y2'): 0}
        interaction_mat = [[A - 8. * B]]

    return Stot, Sz_set, AorB_sym, state_order, interaction_mat


def create_tdo_nn_matrix(lookup_tbl, tdo):
    """
    d轨道和层间O对应的p轨道杂化矩阵
    :param lookup_tbl:包含所有可能的态
    :param tdo:
    :return:out，和apz的杂化哈密顿矩阵coo_matrix
    """
    data = []
    row = []
    col = []

    ph_list = []

    dim = len(lookup_tbl)
    for row_idx in range(dim):
        state = lookup_tbl[row_idx]     # 跳跃前的态
        hole_num = len(state)

        # 遍历这个态的所有空穴
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            z, orb, s = hole
            if orb in ['d3z2r2', 'apz']:       # 只考虑d3z2r2和apz，不考虑dx2y2
                for direction in [-1, 1]:
                    hop_z = z + direction
                    if hop_z in [0, 1, 2]:
                        if hop_z == 1:
                            hop_orb = 'apz'
                        else:
                            hop_orb = 'd3z2r2'
                        hop_hole = (hop_z, hop_orb, s)      # 跳跃后的空穴
                        if hop_hole not in state:       # 跳跃后的空穴不能和其他空穴相同，要满足泡利不相容原理
                            hop_state = list(state)
                            hop_state[hole_idx] = hop_hole      # 将跳跃前的空穴换成跳跃后的空穴
                            inversion = count_inversion(hop_state)      # 计算经过多少次交换后可以使得空穴升序排列（按数字大小和字典序）
                            # 交换偶数次，相位为1，反之为-1
                            if inversion % 2 == 0:
                                ph = 1.0
                            else:
                                ph = -1.0
                            ph_list.append(ph)

                            hop_state = sorted(hop_state)       # 对跳跃后的态按升序排列，规范化
                            hop_state = tuple(hop_state)
                            col_idx = lookup_tbl.index(hop_state)       # 找到跳跃后的态所对应的索引，作为矩阵的列

                            # 在z = 0和1之间的跳跃为-tdo，在z = 1和2之间则为tdo

                            if {z, hop_z} == {0, 1}:
                                element = -tdo * ph
                            else:
                                element = tdo * ph

                            data.append(element)
                            row.append(row_idx)
                            col.append(col_idx)

    def Ni2_Ni0_num(state):
        """
        用来判断在Ni0(z = 0)的空穴数目和在Ni2(z = 2)的空穴数目
        :param state:
        :return: Ni2_num, Ni0_num
        """
        Ni0_num = 0
        Ni2_num = 0
        for hole in state:
            if hole[0] == 0:
                Ni0_num += 1
            if hole[0] == 2:
                Ni2_num += 1
        return Ni0_num , Ni2_num

    num = len(row)
    row_print = []
    col_print = []
    ph_print = []
    data_print = []

    for i in range(num):
        row_idx = row[i]
        col_idx = col[i]
        ph = ph_list[i]
        value = data[i]

        state = lookup_tbl[row_idx]
        hop_state = lookup_tbl[col_idx]
        Ni0_num, Ni2_num = Ni2_Ni0_num(state)
        if Ni0_num == 4 or Ni2_num == 4:
            continue
        Ni0_num , Ni2_num = Ni2_Ni0_num(hop_state)
        if Ni0_num == 4 or Ni2_num == 4:
            continue
        row_print.append(row_idx)
        col_print.append(col_idx)
        ph_print.append(ph)
        data_print.append(value)

    num = len(row_print)
    with open('Tdo_information', 'w') as f:
        for idx in range(num):
            row_idx = row_print[idx]
            col_idx = col_print[idx]
            ph = ph_print[idx]
            value = data_print[idx]
            state = lookup_tbl[row_idx]
            hop_state = lookup_tbl[col_idx]
            state_string = []
            Ni0_num = 0
            Ni2_num = 0
            for hole in state:
                z, orb, s = hole
                if z == 0:
                    Ni0_num += 1
                if z == 2:
                    Ni2_num += 1
                state_string.append(f'({z}, {orb}, {s})')
            if Ni0_num == 4 or Ni2_num == 4:
                continue
            state_string = ', '.join(state_string)
            f.write(state_string + '\n\n')
            state_string = []
            Ni0_num = 0
            Ni2_num = 0
            for hole in hop_state:
                z, orb, s = hole
                if z == 0:
                    Ni0_num += 1
                if z == 2:
                    Ni2_num += 1
                state_string.append(f'({z}, {orb}, {s})')
            if Ni0_num == 4 or Ni2_num == 4:
                continue
            state_string = ', '.join(state_string)
            f.write(state_string + '\n\n')
            f.write(f'ph = {ph}, value = {value}\n\n\n')

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    return out


def create_Esite_matrix(lookup_tbl, A, ed, eo):
    """
    创建Onsite哈密顿矩阵
    :param lookup_tbl:
    :param A:
    :param ed:d3z2r2和dx2y2的在位能，ed = {'d3z2r2': value1, 'dx2y2': value2}
    :param eo:
    :return:out (coo_matrix)
    """
    dim = len(lookup_tbl)
    data = []
    row = []
    col = []
    for i in range(dim):
        state = lookup_tbl[i]
        diag_el = 0
        Ni0 = 0
        Ni2 = 0

        # 遍历态中的所有空穴，求解该态的在位能
        for z, orb, _ in state:
            # d3z2r2, dx2y2, apz的在位能
            if orb in ['d3z2r2', 'dx2y2']:
                diag_el += ed[orb]
            elif orb == 'apz':
                diag_el += eo
            # 统计该态中，在Ni0(z = 0)和Ni2(z = 2)的空穴数目
            if z == 0:
                Ni0 += 1
            elif z == 2:
                Ni2 += 1

        # dn比d8拉高A + abs(n - 8) * A / 2
        if Ni0 != 2:
            diag_el += A + abs(Ni0 - 2) * A / 2.
        if Ni2 != 2:
            diag_el += A + abs(Ni2 - 2) * A / 2.

        data.append(diag_el)
        row.append(i)
        col.append(i)

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    return out


def get_double_occ_list(lookup_tbl):
    """
    找出特定类型的态，这种态在同一个格点上的空穴数目不止一个
    :param lookup_tbl:
    :return:
    """
    d0_double = []
    d0_part = []

    d2_double = []
    d2_part = []
    apz_double = []

    dim = len(lookup_tbl)
    for i in range(dim):
        state = lookup_tbl[i]
        state_type = {}
        for hole_idx, hole in enumerate(state):
            z = hole[0]
            if z in state_type:
                state_type[z] += [hole_idx]
            else:
                state_type[z] = [hole_idx]
        for z, part in state_type.items():
            if z == 0:
                d_num = len(state_type[z])
                if d_num == 2:
                    d0_double.append(i)
                    d0_part.append(part)
            elif z == 2:
                d_num = len(state_type[z])
                if d_num == 2:
                    d2_double.append(i)
                    d2_part.append(part)
            else:
                apz_num = len(state_type[z])
                if apz_num == 2:
                    apz_double.append(i)
    return d0_double, d0_part, d2_double, d2_part, apz_double


def create_interaction_matrix_d(lookup_tbl, d_double, d_part, S_val, Sz_val, A, B, C):
    """
    d8的相互作用
    :param lookup_tbl:
    :param d_double:
    :param d_part:
    :param S_val:
    :param Sz_val:
    :param A:
    :param B:
    :param C:
    :return:
    """
    dim = len(lookup_tbl)
    data = []
    row = []
    col = []
    channels = ['1A1', '1B1', '3B1']
    for sym in channels:
        Stot, Sz_set, AorB, state_order, interaction_mat = get_interaction_mat(A, B, C, sym)
        sym_orbs = state_order.keys()
        for part_idx, tbl_idx in enumerate(d_double):
            count = []
            state = lookup_tbl[tbl_idx]
            hole_idx1, hole_idx2y2 = d_part[part_idx]
            orb1, orb2 = state[hole_idx1][-2], state[hole_idx2y2][-2]
            orb1, orb2 = sorted([orb1, orb2])
            orb12 = (orb1, orb2)
            S12 = S_val[tbl_idx]
            Sz12 = Sz_val[tbl_idx]
            if orb12 not in sym_orbs or S12 != Stot or Sz12 not in Sz_set:
                continue
            mat_idx1 = state_order[orb12]
            for mat_idx2y2, orb34 in enumerate(sym_orbs):
                for s3 in ['dn', 'up']:
                    for s4 in ['dn', 'up']:
                        z1 = state[hole_idx1][0]
                        orb3, orb4 = orb34
                        hole3 = (z1, orb3, s3)
                        hole4 = (z1, orb4, s4)
                        if hole3 == hole4:
                            continue
                        inter_state = list(state)
                        inter_state[hole_idx1], inter_state[hole_idx2y2] = hole3, hole4
                        inter_state = sorted(inter_state)
                        inter_state = tuple(inter_state)
                        inter_idx = lookup_tbl.index(inter_state)
                        if inter_idx in count:
                            continue
                        S34, Sz34 = S_val[inter_idx], Sz_val[inter_idx]
                        if S34 != S12 or Sz34 != Sz12:
                            continue
                        value = interaction_mat[mat_idx1][mat_idx2y2]
                        data.append(value)
                        row.append(tbl_idx)
                        col.append(inter_idx)
                        count.append(inter_idx)

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    return out


def create_interaction_matrix_o(lookup_tbl, apz_double, Uoo):
    """
    层间O(apz)的相互作用
    :param lookup_tbl:
    :param apz_double:
    :param Uoo:
    :return:
    """
    dim = len(lookup_tbl)
    data = []
    row = []
    col = []
    for i in apz_double:
        data.append(Uoo)
        row.append(i)
        col.append(i)

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    return out


def create_singlet_triplet_basis_change_matrix_d_double(lookup_tbl, d_double, d_part):
    """
    创建单态三重态的变换矩阵
    :param lookup_tbl:
    :param d_double:
    :param d_part:
    :return:
    """
    dim = len(lookup_tbl)
    count_list = []
    data = []
    row = []
    col = []
    S_val = np.zeros(dim, dtype=int)
    Sz_val = np.zeros(dim, dtype=int)
    for i in range(dim):
        if i not in d_double:
            data.append(np.sqrt(2))
            row.append(i)
            col.append(i)
    for part_idx, tbl_idx in enumerate(d_double):
        if tbl_idx in count_list:
            continue
        state = lookup_tbl[tbl_idx]
        hole_idx1, hole_idx2 = d_part[part_idx]
        orb1, s1 = state[hole_idx1][-2:]
        orb2, s2 = state[hole_idx2][-2:]
        if s1 == s2:
            data.append(np.sqrt(2))
            row.append(tbl_idx)
            col.append(tbl_idx)
            S_val[tbl_idx] = 1
            if s1 == 'up':
                Sz_val[tbl_idx] = 1
            else:
                Sz_val[tbl_idx] = -1
        else:
            if orb1 != orb2:
                partner_state = [list(hole) for hole in state]
                partner_state[hole_idx1][-1], partner_state[hole_idx2][-1] = (
                    partner_state[hole_idx2][-1], partner_state[hole_idx1][-1]
                )
                inversion = count_inversion(partner_state)
                # ph = 1.0 if inversion % 2 == 0 else -1.0
                ph = 1.0
                partner_state = sorted(partner_state)
                partner_state = tuple(map(tuple, partner_state))
                partner_idx = lookup_tbl.index(partner_state)
                count_list.append(partner_idx)

                data.append(ph)
                row.append(tbl_idx)
                col.append(tbl_idx)

                data.append(-ph)
                row.append(partner_idx)
                col.append(tbl_idx)

                S_val[tbl_idx] = 0
                Sz_val[tbl_idx] = 0


                data.append(1.0)
                row.append(tbl_idx)
                col.append(partner_idx)

                data.append(1.0)
                row.append(partner_idx)
                col.append(partner_idx)

                S_val[partner_idx] = 1
                Sz_val[partner_idx] = 0

            else:
                data.append(np.sqrt(2))
                row.append(tbl_idx)
                col.append(tbl_idx)
                S_val[tbl_idx] = 0
                Sz_val[tbl_idx] = 0

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim)) / np.sqrt(2)
    return out, S_val, Sz_val


def state_classification(state):
    """
    Classify the state into different types
    :param state: state = ((x1, y1, z1, orb1, s1), (x2, y2, z2, orb2, s2)...)
    :return: state_type: string
    """
    Ni0_num = 0
    Ni2_num = 0
    apo_num = 0
    Sz = 0
    for z, orb, s in state:
        if s == 'up':
            Sz += 1 / 2
        else:
            Sz += -1 / 2
        if z == 0:
            Ni0_num += 1
        elif z == 1:
            apo_num += 1
        else:
            Ni2_num += 1
    small_Ni, big_Ni = sorted([Ni0_num, Ni2_num])
    dL_type = []
    if small_Ni != 0:
        str1 = f'd{10 - small_Ni}'
        dL_type.append(str1)
    if apo_num == 1:
        dL_type.append('O')
    elif apo_num == 2:
        dL_type.append('O2')
    if big_Ni != 0:
        str1 = f'd{10 - big_Ni}'
        dL_type.append(str1)
    dL_type = '-'.join(dL_type)
    return dL_type


def get_ground_state(matrix, lookup_tbl, S_Ni0_val, Sz_Ni0_val, S_Ni2_val, Sz_Ni2_val):
    """
    求解哈密顿矩阵
    :param matrix:
    :param lookup_tbl:
    :param S_Ni0_val:
    :param Sz_Ni0_val:
    :param S_Ni2_val:
    :param Sz_Ni2_val:
    :return:
    """
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')
    vals.sort()
    print('lowest eigenvalue of H from np.linalg.eigsh = ')
    print(vals)

    val_num = 2
    degen_idx = [0]
    for _ in range(val_num):
        for idx in range(degen_idx[-1]+1, Neval):
            if abs(vals[idx] - vals[degen_idx[-1]]) > 1e-4:
                degen_idx.append(idx)
                break

    for i in range(val_num):
        print(f'Degeneracy of {i}th state is {degen_idx[i+1] - degen_idx[i]}')
        print(f'val = {vals[degen_idx[i]]}')

        weight_average = np.average(abs(vecs[:, degen_idx[i]:degen_idx[i+1]]) ** 2, axis=1)
        ilead = np.argsort(-weight_average)

        total = 0
        # dL_weight用来记录某种类型(例如d8_d9L)总的weight,
        # dL_istate用来记录某种类型对应的具体态的索引
        # weight_average是各个态的weight
        dL_weight = {}
        dL_istate = {}
        for istate in ilead:
            # state is original state but its orbital info remains after basis change
            weight = weight_average[istate]
            total += weight
            state = lookup_tbl[istate]
            dL_type = state_classification(state)
            if dL_type in dL_weight:
                dL_weight[dL_type] += weight
                dL_istate[dL_type] += [istate]
            else:
                dL_weight[dL_type] = weight
                dL_istate[dL_type] = [istate]
        sorted_dL = sorted(dL_weight, key=dL_weight.get, reverse=True)
        for dL_type in sorted_dL:
            weight = dL_weight[dL_type]
            if weight < 1e-3:
                continue
            print(f'{dL_type}: {weight}\n')
            for istate in dL_istate[dL_type]:
                weight = weight_average[istate]
                if weight < 1e-5:
                    continue
                state = lookup_tbl[istate]
                state_string = []
                for hole in state:
                    z, orb, s = hole
                    state_string.append(f'({z}, {orb}, {s})')
                state_string = ', '.join(state_string)
                print(f'\t{state_string}\n\tS_Ni0 = {S_Ni0_val[istate]}, Sz_Ni0 = {Sz_Ni0_val[istate]}; '
                      f'S_Ni2 = {S_Ni2_val[istate]}, Sz_Ni2 = {Sz_Ni2_val[istate]}, weight = {weight}')
                print(f'\t{vecs[istate, degen_idx[i]:degen_idx[i+1]]}\n')


hole_num = 4
tdo = 1.66
ed = {'d3z2r2': 0.095, 'dx2y2': 0.}
# ed = {'d3z2r2': 0., 'dx2y2': 0.}
eo = 3.24
A = 6.0
B = 0.15
C = 0.58
Uoo = 4.0
Neval = 30
lookup_tbl = create_lookup_tbl()
T_do = create_tdo_nn_matrix(lookup_tbl, tdo)
Esite = create_Esite_matrix(lookup_tbl, A, ed, eo)
H0 = T_do + Esite

d0_double, d0_part, d2_double, d2_part, apz_double = get_double_occ_list(lookup_tbl)
U_Ni0, S_Ni0_val, Sz_Ni0_val = create_singlet_triplet_basis_change_matrix_d_double(lookup_tbl, d0_double, d0_part)
U_Ni2, S_Ni2_val, Sz_Ni2_val = create_singlet_triplet_basis_change_matrix_d_double(lookup_tbl, d2_double, d2_part)
U_Ni0_d = (U_Ni0.conjugate()).transpose()
U_Ni2_d = (U_Ni2.conjugate()).transpose()

Hint_Ni0 = create_interaction_matrix_d(lookup_tbl, d0_double, d0_part,
                                       S_Ni0_val, Sz_Ni0_val, A, B, C)
Hint_Ni2 = create_interaction_matrix_d(lookup_tbl, d2_double, d2_part,
                                       S_Ni2_val, Sz_Ni2_val, A, B, C)
Hint_o = create_interaction_matrix_o(lookup_tbl, apz_double, Uoo)

H0_Ni0 = U_Ni0_d @ H0 @ U_Ni0
H_Ni0 = H0_Ni0 + Hint_Ni0

H_Ni2 = U_Ni2_d @ H_Ni0 @ U_Ni2
H = H_Ni2 + Hint_Ni2 + Hint_o

# H = U_Ni0 @ H @ U_Ni0_d
# H = U_Ni2 @ H @ U_Ni2_d

get_ground_state(H, lookup_tbl, S_Ni0_val, Sz_Ni0_val, S_Ni2_val, Sz_Ni2_val)
