#!/usr/bin/env python3

import re
import sys
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern Roman"]
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.unicode"] = True
mpl.rcParams["text.latex.preamble"] = r"""
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\usepackage{amsmath}
"""

# для начала определяем уровень Ферми
def fermi_level(prefix):
    result = 0
    with open(prefix + ".scf.out", "r") as f:
        data = f.read()
        m = re.search(r"the Fermi energy is\s+(\S+)\s+(\S+)", data)
        if m:
            result = float(m.group(1))
            if m.group(2).lower() == "ry":
                result *= 13.6
    return result

def point_label(point):
    if (point == np.array([0,0,0])).all():
        return r"$$\Gamma$$"
    if (point == np.array([0.5,0.5,0])).all():
        return r"$$F$$"
    if (point == np.array([0,0.5,0.5])).all():
        return r"$$Q$$"
    if (point == np.array([0,0,0.5])).all():
        return r"$$Z$$"
    return r"$$A$$"

# приводим к нормальному виду зонную структуру
# из Ридбергов в эВы и сдвигаем в 0 уровень Ферми
def band_structure(prefix):
    data = np.loadtxt(prefix + ".bands.dat.gnu")
    data[:, 1] *= 13.6
    data[:, 1] -= fermi_level(prefix)
    # добавить определение положения симметричных точек
    points = []
    labels = []
    with open(prefix + ".bands.in") as f:
        flag = False
        pos = 0
        for line in f.readlines():
            if flag:
                if " " not in line.strip():
                    n = int(line.strip())
                else:
                    p = np.fromstring(line, sep=" ")
                    point = p[:3]
                    labels.append(point_label(point))
                    points.append(data[pos, 0])
                    pos += int(p[3])
            if line.startswith("K_POINTS"):
                flag = True
    return data.transpose(), points, labels

# плотность состояний (добавить сглаживание?)
def dos(prefix):
    data = np.loadtxt(prefix + ".dos", usecols=(0, 1))
    data[:, 0] -= fermi_level(prefix)
    return data.transpose()

# спектр поглощения
def epsi(prefix):
    data = np.loadtxt(prefix + "/epsi.dat", skiprows=1)
    return data.transpose()


if __name__ == '__main__':
    prefix = sys.argv[1]
    # строим графики
    # 1. bs + dos
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    (ks, es), points, labels = band_structure(prefix)
    ax1.plot(ks, es, color='#0072bd', linewidth=1)
    ymin, ymax = ax1.get_ylim()
    ax1.set_xlim(min(ks), max(ks))
    ax1.set_xticks(points)
    ax1.set_xticklabels(labels)
    ax1.vlines(points, ymin, ymax)
    ax1.hlines([0], min(ks), max(ks))
    ax1.set_title("Зонная структура")
    ax1.set_ylabel("Энергия, эВ")
    ax1.grid()

    es, ds = dos(prefix)
    ax2.plot(ds, es, color='#d95319', linewidth=0.7)
    ax2.set_ylim([ymin, ymax])
    xmin, xmax = ax2.get_xlim()
    ax2.hlines([0], xmin, xmax, label="Уровень Ферми")
    ax2.set_title("Плотность состояний")
    ax2.grid()

    f.savefig(prefix + "_bandos.png")
    # 2. spectra
    f, ax = plt.subplots()
    es, epsix, epsiy, epsiz = epsi(prefix)
    ax.plot(es, epsix, label=r"$$\varepsilon_{xx}''$$", color='#a2142f')
    ax.plot(es, epsiy, label=r"$$\varepsilon_{yy}''$$", color='#77ac30')
    ax.plot(es, epsiz, label=r"$$\varepsilon_{zz}''$$", color='#0072bd')
    ax.legend()
    ax.grid()
    ax.set_ylabel(r"$$\varepsilon''$$")
    ax.set_xlabel(r"$$\hbar\omega,~\text{эВ}$$")
    f.savefig(prefix + "spectra.png")
