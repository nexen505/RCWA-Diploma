import numpy as np
from numpy import linalg as la
import math
import cmath


def epsilon(index: int, n_ridge: float, n_groove: float, width: float):
    if index == 0:
        return (n_ridge ** 2) * width + (n_groove ** 2) * (1 - width)
    else:
        return (n_ridge ** 2 - n_groove ** 2) * (math.sin(math.pi * index * width)) / (math.pi * index)


def k_xi(k0: float, n: float, i: float, period: float, theta: float):
    lambda0 = 2 * math.pi / k0
    return k0 * (n * math.sin(theta) - i * lambda0 / period)


def k_zi(k0: float, n: float, k: float):
    sq = n ** 2 - (k / k0) ** 2
    if sq >= 0:
        return k0 * cmath.sqrt(sq)
    else:
        return -1j * k0 * cmath.sqrt(-sq)


def delta_function(_i: float):
    return 1 if _i == 0 else 0


class Grating:
    def __init__(self, n_ridge: float, n_groove: float, fillfactor: float, period: float, d_height: float):
        self.n_ridge, self.n_groove, self.fillfactor, self.period, self.d_height = n_ridge, n_groove, fillfactor, period, d_height


def de_r0_te_tm(r, k0: float, n_1: float, theta: float, period: float):
    count = tuple(np.shape(r))[0] // 2
    const = k0 * n_1 * math.cos(theta)
    return r[count] * r[count].conjugate() * (k_zi(k0, n_1, k_xi(k0, n_1, 0, period, theta)) / const).real


def de_t_te(t, k0: float, n_1: float, n_11: float, theta: float, period: float):
    shapes = tuple(np.shape(t))
    count = shapes[0]
    de = np.zeros(count, dtype=float)
    const = k0 * n_1 * math.cos(theta)
    for i in range(count):
        de[i] = (
                t[i] * t[i].conjugate() * (
                k_zi(k0, n_11, k_xi(k0, n_1, i - count // 2, period, theta)) / const).real).real
    return de


def de_t_tm(t, k0: float, n_1: float, n_2: float, theta: float, period: float):
    shapes = tuple(np.shape(t))
    count = shapes[0]
    de = np.zeros(count, dtype=float)
    const = k0 * math.cos(theta) / n_1
    for i in range(count):
        de[i] = (t[i] * t[i].conjugate() *
                 (k_zi(k0, n_2, k_xi(k0, n_1, i - count // 2, period, theta)) / (n_2 ** 2)).real / const).real
    return de


tm_dict = {}
te_dict = {}


def rcwa_te_multilayer_opt(harm: int, theta: float, lambda0: float, n_1: float, n_2: float, period: float,
                           gratings: tuple, optimal: bool = False):
    harm_count = 2 * harm + 1
    k0 = 2 * math.pi / lambda0
    kx = np.zeros((harm_count, harm_count), dtype=complex)
    relation = str(lambda0) + str(theta) + '{:.5f}'.format(lambda0 / period)
    layers_indices = range(len(gratings))
    perm_matrices = {a: np.zeros((harm_count, harm_count), dtype=complex) for a in layers_indices}
    x_matrices = {a: np.zeros((harm_count, harm_count), dtype=complex) for a in layers_indices}
    a_matrices = {a: np.zeros((harm_count, harm_count), dtype=complex) for a in layers_indices}
    for ind in range(harm_count):
        kxi = k_xi(k0, n_1, ind - harm_count // 2, period, theta)
        kx[ind, ind] = kxi / k0
        for l in layers_indices:
            grating = gratings[l]
            for p in range(harm_count):
                perm_matrices[l][ind, p] = epsilon(ind - p, grating.n_ridge, grating.n_groove, grating.fillfactor)

    a_deltai0 = np.zeros(harm_count, dtype=complex)
    for ind in range(harm_count):
        a_deltai0[ind] = delta_function(ind - harm_count // 2)
    a_second_part = 1j * n_1 * math.cos(theta) * a_deltai0
    a_1 = np.concatenate((a_deltai0, a_second_part), axis=0)

    y_1 = np.eye(harm_count, dtype=complex)
    for ind in range(harm_count):
        y_1[ind, ind] = k_zi(k0, n_1, kx[ind, ind] * k0) / k0
    y_11 = np.eye(harm_count, dtype=complex)
    for ind in range(harm_count):
        y_11[ind, ind] = k_zi(k0, n_2, k_xi(k0, n_1, ind - harm_count // 2, period, theta)) / k0

    b = np.concatenate((np.eye(harm_count, dtype=complex), -1j * y_1), axis=0)
    fg = np.concatenate((np.eye(harm_count, dtype=complex), 1j * y_11), axis=0)

    for l in reversed(layers_indices):
        grating = gratings[l]
        e = perm_matrices[l]
        a = np.subtract(la.matrix_power(kx, 2), e)

        if not optimal:
            values, w = la.eig(a)
        else:
            found_dict = None
            if relation in tm_dict:
                found_dict = tm_dict[relation]
                found_dict = found_dict[str(l)] if str(l) in found_dict else None
            if found_dict is None:
                if grating.n_ridge != grating.n_groove and grating.fillfactor != 1:
                    values, w = la.eig(a)
                else:
                    m_size = a.shape[0]
                    values = np.zeros(m_size, dtype=complex)
                    for ind in range(m_size):
                        values[ind] = a[ind, ind]
                    w = np.eye(m_size, dtype=complex)
                    if relation in tm_dict:
                        te_dict[relation][str(l)] = {'vals': values, 'w': w}
                    else:
                        te_dict[relation] = {str(l): {'vals': values, 'w': w}}
            else:
                # print("FOUND!")
                values, w = found_dict['vals'], found_dict['w']

        q_count = values.shape[0]
        e = np.zeros((q_count, q_count), dtype=complex)
        for ind in range(q_count):
            e[ind, ind] = cmath.sqrt(values[ind])
        v = np.dot(w, e)

        x = np.eye(harm_count, dtype=complex)
        for ind in range(harm_count):
            x[ind, ind] = cmath.exp(-k0 * e[ind, ind] * grating.d_height)
        x_matrices[l] = x

        m1 = np.concatenate((w, w), axis=1)
        m2 = np.concatenate((v, -v), axis=1)
        m2 = np.concatenate((m1, m2), axis=0)
        m2 = np.dot(la.inv(m2), fg)
        al, bl = m2[:harm_count, :harm_count], m2[harm_count: 2 * harm_count, :harm_count]
        a_matrices[l] = al
        x_bl_la_inv_al_x = np.dot(x, np.dot(bl, np.dot(la.inv(al), x)))
        fl = np.dot(w, (np.eye(harm_count, dtype=complex) + x_bl_la_inv_al_x))
        gl = np.dot(v, (np.eye(harm_count, dtype=complex) - x_bl_la_inv_al_x))
        fg = np.concatenate((fl, gl), axis=0)

    left_matrix = np.concatenate((-b, fg), axis=1)
    result = la.solve(left_matrix, a_1)
    t1 = result[harm_count: 2 * harm_count]
    for l in layers_indices:
        t1 = np.dot(la.inv(a_matrices[l]), np.dot(x_matrices[l], t1))
    r, t = result[: harm_count], t1
    # r = result[: harm_count]
    return {'r0': de_r0_te_tm(r, k0, n_1, theta, period), 't': de_t_te(t, k0, n_1, n_2, theta, period)}


def rcwa_tm_multilayer_opt(harm: int, theta: float, lambda0: float, n_1: float, n_2: float, period: float,
                           gratings: tuple, optimal: bool = False):
    harm_count = 2 * harm + 1
    k0 = 2 * math.pi / lambda0
    kx = np.zeros((harm_count, harm_count), dtype=complex)
    layers_indices = range(len(gratings))
    relation = str(lambda0) + str(theta) + '{:.5f}'.format(lambda0 / period)
    perm_matrices = {a: np.zeros((harm_count, harm_count), dtype=complex) for a in layers_indices}
    x_matrices = {a: np.zeros((harm_count, harm_count), dtype=complex) for a in layers_indices}
    a_matrices = {a: np.zeros((harm_count, harm_count), dtype=complex) for a in layers_indices}
    for ind in range(harm_count):
        kxi = k_xi(k0, n_1, ind - harm_count // 2, period, theta)
        kx[ind, ind] = kxi / k0
        for l in layers_indices:
            grating = gratings[l]
            for p in range(harm_count):
                perm_matrices[l][ind, p] = epsilon(ind - p, grating.n_ridge, grating.n_groove, grating.fillfactor)

    a_deltai0 = np.zeros(harm_count, dtype=complex)
    for ind in range(harm_count):
        a_deltai0[ind] = delta_function(ind - harm_count // 2)
    a_second_part = 1j * math.cos(theta) * a_deltai0 / n_1
    a_1 = np.concatenate((a_deltai0, a_second_part), axis=0)

    z_1 = np.eye(harm_count, dtype=complex)
    for ind in range(harm_count):
        z_1[ind, ind] = k_zi(k0, n_1, kx[ind, ind] * k0) / (k0 * (n_1 ** 2))
    z_11 = np.eye(harm_count, dtype=complex)
    for ind in range(harm_count):
        z_11[ind, ind] = k_zi(k0, n_2, k_xi(k0, n_1, ind - harm_count // 2, period, theta)) / (k0 * (n_2 ** 2))

    b = np.concatenate((np.eye(harm_count, dtype=complex), -1j * z_1), axis=0)
    fg = np.concatenate((np.eye(harm_count, dtype=complex), 1j * z_11), axis=0)

    for l in reversed(layers_indices):
        grating = gratings[l]
        e = perm_matrices[l]
        e_inv = la.inv(e)
        m1 = np.dot(np.dot(kx, e_inv), kx)
        eb = np.dot(e, np.subtract(m1, np.eye(harm_count, dtype=complex)))

        if not optimal:
            values, w = la.eig(eb)
        else:
            found_dict = None
            if relation in tm_dict:
                found_dict = tm_dict[relation]
                found_dict = found_dict[str(l)] if str(l) in found_dict else None
            if found_dict is None:
                if grating.n_ridge != grating.n_groove and grating.fillfactor != 1:
                    values, w = la.eig(eb)
                else:
                    m_size = eb.shape[0]
                    values = np.zeros(m_size, dtype=complex)
                    for ind in range(m_size):
                        values[ind] = eb[ind, ind]
                    w = np.eye(m_size, dtype=complex)
                    if relation in tm_dict:
                        tm_dict[relation][str(l)] = {'vals': values, 'w': w}
                    else:
                        tm_dict[relation] = {str(l): {'vals': values, 'w': w}}
            else:
                # print("FOUND!")
                values, w = found_dict['vals'], found_dict['w']

        q_count = values.shape[0]
        e = np.zeros((q_count, q_count), dtype=complex)
        for ind in range(q_count):
            e[ind, ind] = cmath.sqrt(values[ind])
        v = np.dot(e_inv, np.dot(w, e))

        x = np.eye(harm_count, dtype=complex)
        for ind in range(harm_count):
            x[ind, ind] = cmath.exp(-k0 * e[ind, ind] * grating.d_height)
        x_matrices[l] = x

        m1 = np.concatenate((w, w), axis=1)
        m2 = np.concatenate((v, -v), axis=1)
        m2 = np.concatenate((m1, m2), axis=0)
        m2 = np.dot(la.inv(m2), fg)
        al, bl = m2[:harm_count, :harm_count], m2[harm_count: 2 * harm_count, :harm_count]
        a_matrices[l] = al
        x_bl_la_inv_al_x = np.dot(x, np.dot(bl, np.dot(la.pinv(al), x)))
        fl = np.dot(w, (np.eye(harm_count, dtype=complex) + x_bl_la_inv_al_x))
        gl = np.dot(v, (np.eye(harm_count, dtype=complex) - x_bl_la_inv_al_x))
        fg = np.concatenate((fl, gl), axis=0)

    left_matrix = np.concatenate((-b, fg), axis=1)
    result = la.solve(left_matrix, a_1)
    t1 = result[harm_count: 2 * harm_count]
    for l in layers_indices:
        t1 = np.dot(la.inv(a_matrices[l]), np.dot(x_matrices[l], t1))
    r, t = result[: harm_count], t1
    return {'r0': de_r0_te_tm(r, k0, n_1, theta, period), 't': de_t_tm(t, k0, n_1, n_2, theta, period)}


