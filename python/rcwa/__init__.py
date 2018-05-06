import cmath
import math
import multiprocessing
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

from rcwa import *
from rcwa.rcwa import Grating
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.titlesize'] = 10


def calculate_0_order(harmonics: int, lambda0: float, layers_count: int, d_w: float, h_d: float, optimal: bool = True):
    period = lambda0 / d_w
    n_ridge = 2.35
    n_groove = 1.0
    _n_1 = 1.0
    _n_11 = 1.48
    _n_layer_1 = 1.38
    _n_layer_2 = 2.35
    theta = 0.00001 * math.pi / 180
    height = h_d * period
    grating = Grating(n_ridge, n_groove, 0.5, period, height)
    l = list()
    l.append(grating)
    for i in range(1, layers_count + 1, 1):
        l.append(Grating(_n_layer_1, 1, 1, period, lambda0 / (4 * _n_layer_1)))
        l.append(Grating(_n_layer_2, 1, 1, period, lambda0 / (4 * _n_layer_2)))
    params = tuple(l)

    opt = rcwa.rcwa_te_multilayer_opt(harmonics, theta, lambda0, _n_1, _n_11, period, params, optimal)
    r_te, t0_te, t1_te = opt['r0'], opt['t'][harmonics], opt['t'][harmonics + 1]
    opt = rcwa.rcwa_tm_multilayer_opt(harmonics, theta, lambda0, _n_1, _n_11, period, params, optimal)
    r_tm, t0_tm, t1_tm = opt['r0'], opt['t'][harmonics], opt['t'][harmonics + 1]

    result_dict = {'te_r0': r_te, 'te_t0': t0_te, 'te_t1': t1_te, 'tm_r0': r_tm, 'tm_t0': t0_tm, 'tm_t1': t1_tm}
    # print(result_dict)
    return result_dict


def draw_te(points_count: int, layers_count: int):
    s = "layers" + str(layers_count)
    print(s)
    x_left = 0.5
    x_right = 2.0
    x = np.linspace(x_left, x_right, points_count)
    z_left = 0
    z_right = 1.5
    z = np.linspace(z_left, z_right, points_count)
    shapes = (z.shape[0], x.shape[0])
    y1 = np.zeros(shapes, dtype=float)
    y2 = np.zeros(shapes, dtype=float)
    y3 = np.zeros(shapes, dtype=float)
    y4 = np.zeros(shapes, dtype=float)
    y5 = np.zeros(shapes, dtype=float)
    y6 = np.zeros(shapes, dtype=float)
    for _x in range(x.shape[0]):
        print(x[_x])
        dicts = Parallel(n_jobs=multiprocessing.cpu_count(), backend="threading", verbose=1)(
            delayed(calculate_0_order)(3, 1064e-9, layers_count, x[_x], z[_z], True) for _z in range(z.shape[0]))
        # dicts = list(range(z.shape[0]))
        # for _z in range(z.shape[0]):
        #     dicts[_z] = calculate_0_order(10, 1064e-9, layers_count, x[_x], z[_z], False)
        for _z in range(z.shape[0]):
            dicts__z_ = dicts[_z]
            y1[_z, _x], y2[_z, _x], y3[_z, _x], y4[_z, _x], y5[_z, _x], y6[_z, _x] = \
                dicts__z_['te_r0'], dicts__z_['tm_r0'], dicts__z_['te_t0'], dicts__z_['tm_t0'], dicts__z_['te_t1'], \
                dicts__z_['tm_t1']

    plt.figure(s)

    plt.subplot(231)
    plt.imshow(y1, interpolation='bilinear',
               origin='lower', extent=[x_left, x_right, z_left, z_right],
               vmax=1.0, vmin=0.0)
    plt.title(r"$R_{0}^{TE}$")
    plt.ylabel("h/d")
    plt.xlabel(r'$\lambda$' + "/d", labelpad=-12)
    plt.colorbar().set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.subplot(232)
    plt.imshow(y2, interpolation='bilinear',
               origin='lower', extent=[x_left, x_right, z_left, z_right],
               vmax=1.0, vmin=0.0)
    plt.title(r"$R_{0}^{TM}$")
    plt.ylabel("h/d")
    plt.xlabel(r'$\lambda$' + "/d", labelpad=-12)
    plt.colorbar().set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    y_tmp = np.zeros(shapes, dtype=float)
    y1_masked = np.zeros(shapes, dtype=int)
    y2_masked = np.zeros(shapes, dtype=int)
    y3_masked = np.zeros(shapes, dtype=float)
    for _x in range(x.shape[0]):
        for _z in range(z.shape[0]):
            a1, a2 = y1[_z, _x], y2[_z, _x]
            y_tmp[_z, _x] = a1 - a2
            y1_masked[_z, _x] = 1 if a1 >= 0.99 else 0
            y2_masked[_z, _x] = 1 if a2 <= 0.8 else 0
            y3_masked[_z, _x] = (a1 - a2) * y1_masked[_z, _x] * y2_masked[_z, _x]
            if y3_masked[_z, _x] != 0:
                print("lambda/d=" + str(x[_x]), "h/d=" + str(z[_z]), y3_masked[_z, _x], sep="-----")

    plt.subplot(233)
    plt.imshow(y_tmp, interpolation='bilinear',
               origin='lower', extent=[x_left, x_right, z_left, z_right],
               vmax=1.0, vmin=-1.0)
    plt.title(r"$R_{0}^{TE} - R_{0}^{TM}$")
    plt.ylabel("h/d")
    plt.xlabel(r'$\lambda$' + "/d", labelpad=-12)
    plt.colorbar().set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    plt.subplot(234)
    plt.imshow(y1_masked, interpolation='nearest',
               origin='lower', extent=[x_left, x_right, z_left, z_right],
               vmax=1.0, vmin=0.0)
    plt.title(r"$R_{0}^{TE}, mask$")
    plt.ylabel("h/d")
    plt.xlabel(r'$\lambda$' + "/d")
    plt.colorbar().set_ticks([0, 1])

    plt.subplot(235)
    plt.imshow(y2_masked, interpolation='nearest',
               origin='lower', extent=[x_left, x_right, z_left, z_right], aspect='equal',
               vmax=1.0, vmin=0.0)
    plt.title(r"$R_{0}^{TM}, mask$")
    plt.ylabel("h/d")
    plt.xlabel(r'$\lambda$' + "/d")
    plt.colorbar().set_ticks([0, 1])

    plt.subplot(236)
    plt.imshow(y3_masked, interpolation='nearest',
               origin='lower', extent=[x_left, x_right, z_left, z_right], aspect='equal',
               vmax=1.0, vmin=-1.0)
    plt.title(r"$R_{0}^{TE} - R_{0}^{TM}, mask$")
    plt.ylabel("h/d")
    plt.xlabel(r'$\lambda$' + "/d")
    plt.colorbar().set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    plt.figure(s + " with orders")

    plt.subplot(231)
    plt.imshow(y1, interpolation='bilinear',
               origin='lower', extent=[x_left, x_right, z_left, z_right],
               vmax=1.0, vmin=0.0)
    plt.title(r"$R_{0}^{TE}$")
    plt.ylabel("h/d")
    plt.xlabel(r'$\lambda$' + "/d", labelpad=-12)
    plt.colorbar().set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.subplot(234)
    plt.imshow(y2, interpolation='bilinear',
               origin='lower', extent=[x_left, x_right, z_left, z_right],
               vmax=1.0, vmin=0.0)
    plt.title(r"$R_{0}^{TM}$")
    plt.ylabel("h/d")
    plt.xlabel(r'$\lambda$' + "/d", labelpad=-12)
    plt.colorbar().set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.subplot(232)
    plt.imshow(y3, interpolation='bilinear',
               origin='lower', extent=[x_left, x_right, z_left, z_right],
               vmax=np.amax(y3), vmin=np.amin(y3))
    plt.title(r"$T_{0}^{TE}$")
    plt.ylabel("h/d")
    plt.xlabel(r'$\lambda$' + "/d", labelpad=-12)
    plt.colorbar()

    plt.subplot(235)
    plt.imshow(y4, interpolation='bilinear',
               origin='lower', extent=[x_left, x_right, z_left, z_right],
               vmax=np.amax(y4), vmin=np.amin(y4))
    plt.title(r"$T_{0}^{TM}$")
    plt.ylabel("h/d")
    plt.xlabel(r'$\lambda$' + "/d")
    plt.colorbar()

    plt.subplot(233)
    plt.imshow(y5, interpolation='bilinear',
               origin='lower', extent=[x_left, x_right, z_left, z_right],
               vmax=np.amax(y5), vmin=np.amin(y5))
    plt.title(r"$T_{1}^{TE}$")
    plt.ylabel("h/d")
    plt.xlabel(r'$\lambda$' + "/d", labelpad=-12)
    plt.colorbar()

    plt.subplot(236)
    plt.imshow(y6, interpolation='bilinear',
               origin='lower', extent=[x_left, x_right, z_left, z_right],
               vmax=np.amax(y6), vmin=np.amin(y6))
    plt.title(r"$T_{1}^{TM}$")
    plt.ylabel("h/d")
    plt.xlabel(r'$\lambda$' + "/d")
    plt.colorbar()


def draw_structure_by_height():
    harmonics = 10
    lambda0 = 1064e-9
    d_w = 1.0325
    h_d = 0.615
    # h = 633.762709e-9
    layers_count = 7
    period = lambda0 / d_w
    n_ridge = 2.35
    n_groove = 1.0
    _n_1 = 1.0
    _n_11 = 1.48
    _n_layer_1 = 1.38
    _n_layer_2 = 2.35
    theta = 0.0000001 * math.pi / 180
    step = 0.0005
    d = np.arange(0.5, 0.8 + step, step)
    _r_values = list()
    _t_values = list()
    has_r = True
    has_t = True
    for height in d:
        grating = Grating(n_ridge, n_groove, 0.5, period, height * 1e-6)
        l = list()
        l.append(grating)
        for i in range(1, layers_count + 1, 1):
            l.append(Grating(_n_layer_1, 1, 1, period, lambda0 / (4 * _n_layer_1)))
            l.append(Grating(_n_layer_2, 1, 1, period, lambda0 / (4 * _n_layer_2)))
        params = tuple(l)
        v1 = rcwa.rcwa_te_multilayer_opt(harmonics, theta, lambda0, _n_1, _n_11, period, params)['r0']
        v2 = rcwa.rcwa_tm_multilayer_opt(harmonics, theta, lambda0, _n_1, _n_11, period, params)['r0']
        print(height, v1, v2, sep="---")
        if has_r:
            _r_values.append(v1)
        if has_t:
            _t_values.append(v2)

    plt.figure("Height")
    if has_r:
        line1, = plt.plot(d, _r_values, label=r"$R_{0}^{TE}$", color="red", linewidth=1.0, linestyle="-")
        plt.ylabel("Отражение")
        first_legend = plt.legend(handles=[line1], loc=2)
        ax = plt.gca().add_artist(first_legend)
    if has_t:
        line2, = plt.plot(d, _t_values, label=r"$R_{0}^{TM}$", color="blue", linewidth=1.0, linestyle="-")
        plt.ylabel("Пропускание")
        plt.legend(handles=[line2], loc=3)
    if has_r and has_t:
        plt.ylabel("Интенсивность")
        # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Высота дифракционной решетки (мкм)")


def draw_structure_by_fillfactor():
    harmonics = 10
    lambda0 = 1064e-9
    d_w = 1.0325
    h_d = 0.615
    layers_count = 7
    period = lambda0 / d_w
    height = period * h_d
    n_ridge = 2.35
    n_groove = 1.0
    _n_1 = 1.0
    _n_11 = 1.48
    _n_layer_1 = 1.38
    _n_layer_2 = 2.35
    theta = 0.00001 * math.pi / 180
    d = np.linspace(0, 1, 601)
    _r_values = list()
    _t_values = list()
    has_r = True
    has_t = True
    l = list()
    for i in range(layers_count):
        l.append(Grating(_n_layer_1, _n_layer_1, 1, period, lambda0 / (4 * _n_layer_1)))
        l.append(Grating(_n_layer_2, _n_layer_2, 1, period, lambda0 / (4 * _n_layer_2)))
    for fillfactor in d:
        grating = Grating(n_ridge, n_groove, fillfactor, period, height)
        l.insert(0, grating)
        params = tuple(l)
        v1 = rcwa.rcwa_te_multilayer_opt(harmonics, theta, lambda0, _n_1, _n_11, period, params)['r0']
        v2 = rcwa.rcwa_tm_multilayer_opt(harmonics, theta, lambda0, _n_1, _n_11, period, params)['r0']
        print(fillfactor, v1, v2, sep="---")
        if has_r:
            _r_values.append(v1)
        if has_t:
            _t_values.append(v2)
        l.pop(0)

    plt.figure("Fillfactor")
    if has_r:
        line1, = plt.plot(d, _r_values, label=r"$R_{0}^{TE}$", color="red", linewidth=1.0, linestyle="-", scaley=1e-11)
        plt.ylabel("Отражение")
        first_legend = plt.legend(handles=[line1], loc=2)
        plt.gca().add_artist(first_legend)
    if has_t:
        line2, = plt.plot(d, _t_values, label=r"$R_{0}^{TM}$", color="blue", linewidth=1.0, linestyle="-", scaley=1e-11)
        plt.ylabel("Пропускание")
        plt.legend(handles=[line2], loc=3)
    if has_r and has_t:
        plt.ylabel("Интенсивность")
        # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Отношение ширины ступеньки к периоду")


def draw_forbidden_zone():
    lambda0 = 1064e-9
    d_w = 1.0325
    period = lambda0 / d_w
    _te_values = list()
    _tm_values = list()
    d = np.linspace(0, 2.2, 300)
    lambda0 = 1064e-9
    k0 = 2 * math.pi / lambda0
    _n_layer_1 = 1.38
    _n_layer_2 = 2.35

    h1 = lambda0 / (4 * _n_layer_1)
    h2 = lambda0 / (4 * _n_layer_2)
    for n in d:
        k_z_1 = cmath.sqrt((k0 * _n_layer_1) ** 2 - (k0 * n) ** 2)
        k_z_2 = cmath.sqrt((k0 * _n_layer_2) ** 2 - (k0 * n) ** 2)
        v1 = abs(cmath.cos(k_z_1 * h1) * cmath.cos(k_z_2 * h2) - cmath.sin(k_z_1 * h1) * cmath.sin(k_z_2 * h2) * (
                k_z_1 ** 2 + k_z_2 ** 2) / (2 * k_z_1 * k_z_2))
        v2 = abs(cmath.cos(k_z_1 * h1) * cmath.cos(k_z_2 * h2) - cmath.sin(k_z_1 * h1) * cmath.sin(k_z_2 * h2) * (
                (k_z_1 / (_n_layer_1 ** 2)) ** 2 + (k_z_2 / (_n_layer_2 ** 2)) ** 2) / (
                         2 * k_z_1 * k_z_2 / ((_n_layer_1 * _n_layer_2) ** 2)))
        _te_values.append(v1)
        _tm_values.append(v2)
    plt.figure("Forbidden zone")
    line1, = plt.plot(d, _te_values, label="TE", color="red", linewidth=1.0, linestyle="-")
    first_legend = plt.legend(handles=[line1], loc=2)
    plt.gca().add_artist(first_legend)
    line2, = plt.plot(d, _tm_values, label="TM", color="blue", linewidth=1.0, linestyle="-")
    plt.legend(handles=[line2], loc=3)
    plt.axvline(x=lambda0 / period, linewidth=0.5, linestyle="--")
    plt.axhline(y=1, linewidth=0.5, linestyle="--")
    plt.ylabel(r"$|cos(\~{k}d)|$")
    plt.xlabel(r"$n_{eff}$")


def draw_structure_by_wavelength():
    harmonics = 10
    lambda0 = 1064e-9
    d_w = 1.0325
    h_d = 0.615
    # h = 633.762709e-9
    layers_count = 7
    period = lambda0 / d_w
    height = period * h_d
    _n_1 = 1.0
    _n_11 = 1.48
    _n_layer_1 = 1.38
    _n_layer_2 = 2.35
    theta = 0.0000001 * math.pi / 180
    d = range(975, 1176, 1)
    _r_values = list()
    _t_values = list()
    l = list()
    for i in range(layers_count):
        l.append(Grating(_n_layer_1, _n_layer_1, 1, period, lambda0 / (4 * _n_layer_1)))
        l.append(Grating(_n_layer_2, _n_layer_2, 1, period, lambda0 / (4 * _n_layer_2)))
    for wavelength in d:
        grating = Grating(2.35, 1.0, 0.5, period, height)
        l.insert(0, grating)
        params = tuple(l)
        v1 = rcwa.rcwa_te_multilayer_opt(harmonics, theta, wavelength * 1e-9, _n_1, _n_11, period, params)['r0']
        v2 = rcwa.rcwa_tm_multilayer_opt(harmonics, theta, wavelength * 1e-9, _n_1, _n_11, period, params)['r0']
        print(wavelength, v1, v2, sep="---")
        _r_values.append(v1)
        _t_values.append(v2)
        l.pop(0)

    plt.figure("Wavelength")
    line1, = plt.plot(d, _r_values, label=r"$R_{0}^{TE}$", color="red", linewidth=1.0, linestyle="-")
    first_legend = plt.legend(handles=[line1], loc=2)
    plt.gca().add_artist(first_legend)
    line2, = plt.plot(d, _t_values, label=r"$R_{0}^{TM}$", color="blue", linewidth=1.0, linestyle="-")
    plt.legend(handles=[line2], loc=3)
    plt.ylabel("Интенсивность")
    plt.xlabel("Длина волны (нм)")


def draw_structure_by_period():
    harmonics = 10
    lambda0 = 1064e-9
    d_w = 1.0325
    h_d = 0.615
    # h = 633.762709e-9
    layers_count = 7
    period = lambda0 / d_w
    height = period * h_d
    _n_1 = 1.0
    _n_11 = 1.48
    _n_layer_1 = 1.38
    _n_layer_2 = 2.35
    theta = 0.0000001 * math.pi / 180
    period_mean = period * 1e9
    d = np.linspace(period_mean - 200, period_mean + 200, 500)
    _r_values = list()
    _t_values = list()
    l = list()
    for i in range(layers_count):
        l.append(Grating(_n_layer_1, _n_layer_1, 1, period, lambda0 / (4 * _n_layer_1)))
        l.append(Grating(_n_layer_2, _n_layer_2, 1, period, lambda0 / (4 * _n_layer_2)))
    for _period in d:
        grating = Grating(2.35, 1.0, 0.5, _period * 1e-9, height)
        l.insert(0, grating)
        params = tuple(l)
        v1 = rcwa.rcwa_te_multilayer_opt(harmonics, theta, lambda0, _n_1, _n_11, _period * 1e-9, params)['r0']
        v2 = rcwa.rcwa_tm_multilayer_opt(harmonics, theta, lambda0, _n_1, _n_11, _period * 1e-9, params)['r0']
        print(_period, v1, v2, sep="---")
        _r_values.append(v1)
        _t_values.append(v2)
        l.pop(0)

    plt.figure("Period")
    line1, = plt.plot(d, _r_values, label=r"$R_{0}^{TE}$", color="red", linewidth=1.0, linestyle="-")
    first_legend = plt.legend(handles=[line1], loc=2)
    plt.gca().add_artist(first_legend)
    line2, = plt.plot(d, _t_values, label=r"$R_{0}^{TM}$", color="blue", linewidth=1.0, linestyle="-")
    plt.legend(handles=[line2], loc=3)
    plt.ylabel("Интенсивность")
    plt.xlabel("Период решетки (нм)")


# draw_structure_by_height()
# draw_structure_by_fillfactor()
# draw_structure_by_wavelength()
# draw_structure_by_period()
# draw_forbidden_zone()

for i in range(7, 8):
    start_time = datetime.now()
    print("Start time: " + start_time.strftime('%Y-%m-%d %H:%M:%S'))
    draw_te(201, i)
    end_time = datetime.now()
    print("End time: " + end_time.strftime('%Y-%m-%d %H:%M:%S'))
    print("Calculation time: " + str((end_time - start_time).seconds))
    plt.show(block=False)
    plt.pause(3)

plt.show()
