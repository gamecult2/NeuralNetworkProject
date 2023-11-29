import os
import numpy as np
import openseespy.opensees as ops
import opsvis as opsv
#import vfo.vfo as vfo
import matplotlib.pyplot as plt
import NewGeneratePeaks

# Nodal Coordinate
ops.wipe()
ops.model('Basic', '-ndm', 2, '-ndf', 3)
pielen = 88.58
piewid = 129.92
unblED = 15.75
anclED = 31.89
anclPT = 23.62
pieH = 21.26
pieB = 15.75
numENT = 9
numED = 4
numPT = 2
locED = [-8.86, -5.31, 5.31, 8.86]
locPT = [-3.94, 3.94]
t = pieB
# base = 0
# pier1 = 0
# Top = 1000
# pier2 = 2000
baspier1 = 0
toppier1 = 1000
baspier2 = 2000
toppier2 = 3000
# Create nodes
# ------------------------------------------------------------------------
# node nodeId xCrd yCrd..
# ------------------------------------------------------------------------

Impact_coordinate_Y = 600
#                              1   2   3   4   5   6   7   8  9   10  11  12  13  14  15  16  17  18  19  20  21   22   23
element_length = np.array([0, 15.75, 15.75, 15.75, 15.75, 25.58])
# element_length = np.array([0, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 400, 400, 450])
#                              1   2   3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21    22    23    24
Nodes_coordinate_Y = np.array([0, 15.75, 31.5, 47.25, 63.00, 88.58])
# Nodes_coordinate_Y = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1400, 1800, 2250])

# Pier 1
for i in range(1, len(element_length) - 1):
    ops.node(i + 1, 0, int(Nodes_coordinate_Y[i]))
    print('node', i + 1, 0, int(Nodes_coordinate_Y[i]))
# Pier 2
for i in range(1, len(element_length) - 1):
    ops.node(i + 1 + len(element_length), piewid, int(Nodes_coordinate_Y[i]))
    print('node', i + 1 + len(element_length), piewid, int(Nodes_coordinate_Y[i]))
# Bent Beam
ops.node(2 * len(element_length) + 1, 0, pielen + 11.81)
ops.node(2 * len(element_length) + 2, piewid, pielen + 11.81)
print('node', 2 * len(element_length) + 1, 0, pielen + 11.81)
print('node', 2 * len(element_length) + 2, piewid, pielen + 11.81)
# Rocking Nodes (Pier 1 and Pier 2)

# Pier 1 top
# [1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004]
#   |     |     |     |     |     |     |     |     |
# [1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994]

# Pier 1 base
# [996,  997,  998,  999,  1000, 1001, 1002, 1003, 1004]
#   |     |     |     |     |     |     |     |     |
# [986,  987,  988,  989,  990,  991,  992,  993,  994]

# Pier 2 top
# [3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004]
#   |     |     |     |     |     |     |     |     |
# [3986, 3987, 3988, 3989, 3990, 3991, 3992, 3993, 3994]

# Pier 2 base
# [2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004]
#   |     |     |     |     |     |     |     |     |
# [2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994]

bas1botnodes = []
bas1topnodes = []
top1botnodes = []
top1topnodes = []
bas2botnodes = []
bas2topnodes = []
top2botnodes = []
top2topnodes = []
for i in range(0, numENT):
    bas1botnodes.append(int(986 + baspier1 + i))
    bas1topnodes.append(int(986 + baspier1 + numENT + 1 + i))
    top1botnodes.append(int(986 + toppier1 + i))
    top1topnodes.append(int(986 + toppier1 + numENT + 1 + i))
    bas2botnodes.append(int(986 + baspier2 + i))
    bas2topnodes.append(int(986 + baspier2 + numENT + 1 + i))
    top2botnodes.append(int(986 + toppier2 + i))
    top2topnodes.append(int(986 + toppier2 + numENT + 1 + i))
# print(bas1botnode)
# print(bas1topnode)
# print(top1botnode)
# print(top1topnode)
# print(bas2botnode)
# print(bas2topnode)
# print(top2botnode)
# print(top2topnode)
for i in range(0, numENT):
    ops.node(bas1botnodes[i], -pieH / 2 + i * pieH / (numENT - 1), 0)
    ops.node(bas1topnodes[i], -pieH / 2 + i * pieH / (numENT - 1), 0)
    ops.node(top1botnodes[i], -pieH / 2 + i * pieH / (numENT - 1), pielen)
    ops.node(top1topnodes[i], -pieH / 2 + i * pieH / (numENT - 1), pielen)
    ops.node(bas2botnodes[i], piewid+ (-pieH) / 2 + i * pieH / (numENT - 1), 0)
    ops.node(bas2topnodes[i], piewid+ (-pieH) / 2 + i * pieH / (numENT - 1), 0)
    ops.node(top2botnodes[i], piewid+ (-pieH) / 2 + i * pieH / (numENT - 1), pielen)
    ops.node(top2topnodes[i], piewid+ (-pieH) / 2 + i * pieH / (numENT - 1), pielen)
    print('node', bas1botnodes[i], -pieH / 2 + i * pieH / (numENT - 1), 0)
    print('node', bas1topnodes[i], -pieH / 2 + i * pieH / (numENT - 1), 0)
    print('node', top1botnodes[i], -pieH / 2 + i * pieH / (numENT - 1), pielen)
    print('node', top1topnodes[i], -pieH / 2 + i * pieH / (numENT - 1), pielen)
    print('node', bas2botnodes[i], piewid+ (-pieH) / 2 + i * pieH / (numENT - 1), 0)
    print('node', bas2topnodes[i], piewid+ (-pieH) / 2 + i * pieH / (numENT - 1), 0)
    print('node', top2botnodes[i], piewid+ (-pieH) / 2 + i * pieH / (numENT - 1), pielen)
    print('node', top2topnodes[i], piewid+ (-pieH) / 2 + i * pieH / (numENT - 1), pielen)
pie1node = []
pie2node = []
pie1node.append(bas1topnodes[int((numENT - 1) / 2)])
pie2node.append(bas2topnodes[int((numENT - 1) / 2)])
for i in range(1, len(element_length) - 1):
    pie1node.append(i + 1)
    pie2node.append(i + 1 + len(element_length))
pie1node.append(top1botnodes[int((numENT - 1) / 2)])
pie2node.append(top2botnodes[int((numENT - 1) / 2)])
print(pie1node)
print(pie2node)


# Energy Dissipating Rebars Node
for i in range(0, 2):
    for j in range(0, numED):
        ops.node(100 + numED * i + j, locED[j] + i * piewid, -anclED)
        ops.node(200 + numED * i + j, locED[j] + i * piewid, unblED)
        print('node', 100 + numED * i + j, locED[j] + i * piewid, -anclED, ';')
        print('node', 200 + numED * i + j, locED[j] + i * piewid, unblED, ';')
# PT Node
for i in range(0, 2):
    for j in range(0, numPT):
        ops.node(300 + numPT * i + j, locPT[j] + i * piewid, -anclPT)
        ops.node(400 + numPT * i + j, locPT[j] + i * piewid, pielen)
        print('node', 300 + numPT * i + j, locPT[j] + i * piewid, -anclPT)
        print('node', 400 + numPT * i + j, locPT[j] + i * piewid, pielen)

# ------------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------------
# Constraints of base spring bottom nodes of pier1 and pier2
for i in range(0, len(bas1botnodes)):
    ops.fix(bas1botnodes[i], 1, 1, 1)
    ops.fix(bas2botnodes[i], 1, 1, 1)
    print('fix', bas1botnodes[i], 1, 1, 1)
    print('fix', bas2botnodes[i], 1, 1, 1)
# Constraints of base spring top nodes of pier1 and pier2
for i in range(0, 1):
    ops.fix(bas1topnodes[i], 1, 0, 0)
    ops.fix(bas2topnodes[i], 1, 0, 0)
    print('fix', bas1topnodes[i], 1, 0, 0)
    print('fix', bas2topnodes[i], 1, 0, 0)
# Constraints of ED truss bottom nodes of pier1 and pier2
for i in range(0, 2):
    for j in range(0, numED):
        ops.fix(100 + numED * i + j, 1, 1, 1)
        print('fix', 100 + numED * i + j, 1, 1, 1)
# Constraints of PT truss bottom nodes of pier1 and pier2
for i in range(0, 2):
    for j in range(0, numPT):
        ops.fix(300 + numPT * i + j, 1, 1, 1)
        print('fix', 300 + numPT * i + j, 1, 1, 1)
# EqualDOF of top spring bottom and top nodes of pier1 and pier2
for i in range(0, len(top1botnodes)):
    ops.equalDOF(top1topnodes[i], top1botnodes[i], 1)
    ops.equalDOF(top2topnodes[i], top2botnodes[i], 1)
    print('equalDOF', top1topnodes[i], top1botnodes[i], 1)
    print('equalDOF', top2topnodes[i], top2botnodes[i], 1)
# ------------------------------------------------------------------------
# Define uniaxial materials for 2D RC Panel Constitutive Model (FSAM)
# ------------------------------------------------------------------------
# Steel X
fyx = 61.6410    # yield strength of transverse reinforcement in ksi
bx = 0.01   # strain hardening coefficient of transverse reinforcement
# Steel Y
fyY = 65.99217     # yield strength of longitudinal reinforcement in ksi
by = 0.01   # strain hardening coefficient of longitudinal reinforcement
# Steel misc
Esy = 29000  # Young's modulus (199947.9615MPa)
Esx = 29000  # Young's modulus (199947.9615MPa)
R0 = 20      # Initial value of curvature parameter
A1 = 0.925   # Curvature degradation parameter
A2 = 0.15    # Curvature degradation parameter
# Steel ED
IDED = 3
fyED = 42.2059
EED = 29000
b = 0.01
# Steel PT
IDPT = 4
fyPT = 232.0604
EPT = 29000
a1 = 0
a2 = 1
a3 = 0
a4 = 1
sigInit = 109.5035
# Build SteelMPF material
ops.uniaxialMaterial('SteelMPF', 1, fyx, fyx, Esx, bx, bx, R0, A1, A2)  #Steel X
ops.uniaxialMaterial('SteelMPF', 2, fyY, fyY, Esy, by, by, R0, A1, A2)  #Steel Y
ops.uniaxialMaterial('Steel02', IDED, fyED, EED, b, R0, A1, A2)  # Steel ED
ops.uniaxialMaterial('Steel02', IDPT, fyPT, EPT, b, R0, A1, A2, a1, a2, a3, a4, sigInit)   # Steel PT
# Concrete
# unconfined concrete
fpc = 5.578        # peak compressive stress(5.578)
ec0 = -0.002161  # strain at peak compressive stress
ft = 0.56015    # peak tensile stress
et = 0.000238    # strain at peak tensile stress
Ec = 4700.083    # Young's modulus
xcrnu = 1.035    # cracking strain (compression)
xcrp = 10000     # cracking strain (tension)
ru = 5.5375         # shape parameter (compression)
rt = 1.2         # shape parameter (tension)
# confined concrete
fpcc = 7.2516      # peak compressive stress
ec0c = -0.00231  # strain at peak compressive stress
ftc = 0.6387   # peak tensile stress
etc = 0.0002463   # strain at peak tensile stress
Ecc = 5186.0219  # Young's modulus
xcrnc = 1.035    # cracking strain (compression)
rc = 7.7687        # shape parameter (compression)
# Build ConcreteCM material
ops.uniaxialMaterial('ConcreteCM', 5, -fpc, ec0, Ec, ru, xcrnu, ft, et, rt, xcrp, '-GapClose', 0)  # unconfined concrete
ops.uniaxialMaterial('ConcreteCM', 6, -fpcc, ec0c, Ecc, rc, xcrnc, ftc, etc, rt, xcrp, '-GapClose', 0)  # confined concrete
# ---------------------------------------
#  Define 2D RC Panel Material (FSAM)
# ---------------------------------------

# Reinforcing ratios
rouX = 0.002   # Reinforcing ratio of transverse rebar
rouYc = 0.04744  # Reinforcing ratio of cover
rouY1 = 0.04744  # Reinforcing ratio of longitudinal rebar(12f20: different from Han 2019)
nu = 0.2         # Friction coefficient
alfadow = 0.012  # Dowel action stiffness parameter

# Build ndMaterial FSAM
ops.nDMaterial('FSAM', 7, 0.0, 1, 2, 5, rouX, rouYc, nu, alfadow)
ops.nDMaterial('FSAM', 8, 0.0, 1, 2, 6, rouX, rouY1, nu, alfadow)
# Define ENT Material
IDENT = 9
EENT = 2 * 25.4 * 4700.08368 * pieH * pieB / (pielen * numENT)
# print(EENT)

# print(EENT)
ops.uniaxialMaterial('ENT', IDENT, EENT)

# ------------------------------
#  Define SFI_MVLEM elements
# ------------------------------
t = 15.75
for i in range(0, len(element_length) - 1):
    ops.element('SFI_MVLEM', i + 101, pie1node[i], pie1node[i + 1], 10, 0.4, '-thick', t, t, t, t, t, t, t, t, t, t, '-width', 2.79, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 2.79, '-mat', 7, 8, 8, 8, 8, 8, 8, 8, 8, 7)
    ops.element('SFI_MVLEM', i + 151, pie2node[i], pie2node[i + 1], 10, 0.4, '-thick', t, t, t, t, t, t, t, t, t, t, '-width', 2.79, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 2.79, '-mat', 7, 8, 8, 8, 8, 8, 8, 8, 8, 7)
    print('element', 'SFI_MVLEM', i + 101, pie1node[i], pie1node[i + 1], 7, 0.4, '-thick', t, t, t, t, t, t, t, '-width', 50, 88, 88, 88, 88, 88, 50, '-mat', 7, 8, 8, 8, 8, 8, 7)
    print('element', 'SFI_MVLEM', i + 151, pie2node[i], pie2node[i + 1], 7, 0.4, '-thick', t, t, t, t, t, t, t, '-width', 50, 88, 88, 88, 88, 88, 50, '-mat', 7, 8, 8, 8, 8, 8, 7)


# Define Zero-Length Element
for i in range(0, numENT):
    ops.element('zeroLength', i + 601, bas1botnodes[i], bas1topnodes[i], '-mat', IDENT, '-dir', 2)
    ops.element('zeroLength', i + 701, top1botnodes[i], top1topnodes[i], '-mat', IDENT, '-dir', 2)
    ops.element('zeroLength', i + 801, bas2botnodes[i], bas2topnodes[i], '-mat', IDENT, '-dir', 2)
    ops.element('zeroLength', i + 901, top2botnodes[i], top2topnodes[i], '-mat', IDENT, '-dir', 2)
    print('zeroLength', i + 601, bas1botnodes[i], bas1topnodes[i], '-mat', IDENT, '-dir', 2)
    print('zeroLength', i + 701, top1botnodes[i], top1topnodes[i], '-mat', IDENT, '-dir', 2)
    print('zeroLength', i + 801, bas2botnodes[i], bas2topnodes[i], '-mat', IDENT, '-dir', 2)
    print('zeroLength', i + 901, top2botnodes[i], top2topnodes[i], '-mat', IDENT, '-dir', 2)
geomTransfTag_PDelta = 1
ops.geomTransf('PDelta', geomTransfTag_PDelta)
for i in range(0, numENT - 1):
    ops.element('elasticBeamColumn', i + 1601, bas1topnodes[i], bas1topnodes[i + 1], 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', i + 1701, top1botnodes[i], top1botnodes[i + 1], 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', i + 1711, top1topnodes[i], top1topnodes[i + 1], 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', i + 1801, bas2topnodes[i], bas2topnodes[i + 1], 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', i + 1901, top2botnodes[i], top2botnodes[i + 1], 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', i + 1911, top2topnodes[i], top2topnodes[i + 1], 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
    print('elasticBeamColumn', i + 1601, bas1topnodes[i], bas1topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('elasticBeamColumn', i + 1701, top1botnodes[i], top1botnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('elasticBeamColumn', i + 1711, top1topnodes[i], top1topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('elasticBeamColumn', i + 1801, bas2topnodes[i], bas2topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('elasticBeamColumn', i + 1901, top2botnodes[i], top2botnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('elasticBeamColumn', i + 1911, top2topnodes[i], top2topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
# Define ED elements
ops.element('truss', 201, 100, 200, 0.487 * 4, IDED)
ops.element('truss', 202, 101, 201, 0.487 * 4, IDED)
ops.element('truss', 203, 102, 202, 0.487 * 2, IDED)
ops.element('truss', 204, 103, 203, 0.487 * 2, IDED)
ops.element('elasticBeamColumn', 211, 200, 201, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
ops.element('elasticBeamColumn', 212, 201, int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
ops.element('elasticBeamColumn', 213, int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1, 202, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
ops.element('elasticBeamColumn', 214, 202, 203, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
print('element', 'truss', 201, 100, 200, 0.487 * 4, IDED)
print('element', 'truss', 202, 101, 201, 0.487 * 4, IDED)
print('element', 'truss', 203, 102, 202, 0.487 * 2, IDED)
print('element', 'truss', 204, 103, 203, 0.487 * 2, IDED)
print('element', 'elasticBeamColumn', 211, 200, 201, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
print('element', 'elasticBeamColumn', 212, 201, int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
print('element', 'elasticBeamColumn', 213, int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1, 202, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
print('element', 'elasticBeamColumn', 214, 202, 203, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
ops.element('truss', 221, 104, 204, 0.487 * 4, IDED)
ops.element('truss', 222, 105, 205, 0.487 * 4, IDED)
ops.element('truss', 223, 106, 206, 0.487 * 2, IDED)
ops.element('truss', 224, 107, 207, 0.487 * 2, IDED)
ops.element('elasticBeamColumn', 231, 204, 205, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
ops.element('elasticBeamColumn', 232, 205, int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1 + len(element_length), 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
ops.element('elasticBeamColumn', 233, int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1 + len(element_length), 206, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
ops.element('elasticBeamColumn', 234, 206, 207, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
print('element', 'truss', 221, 104, 204, 0.487 * 4, IDED)
print('element', 'truss', 222, 105, 205, 0.487 * 4, IDED)
print('element', 'truss', 223, 106, 206, 0.487 * 2, IDED)
print('element', 'truss', 224, 107, 207, 0.487 * 2, IDED)
print('element', 'elasticBeamColumn', 231, 204, 205, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
print('element', 'elasticBeamColumn', 232, 205, int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1 + len(element_length), 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
print('element', 'elasticBeamColumn', 233, int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1 + len(element_length), 206, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
print('element', 'elasticBeamColumn', 234, 206, 207, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
# Define PT elements
ops.element('truss', 301, 300, 400, 0.434, IDPT)
ops.element('truss', 302, 301, 401, 0.434, IDPT)
ops.element('elasticBeamColumn', 311, 400, pie1node[-1], 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
ops.element('elasticBeamColumn', 312, pie1node[-1], 401, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
ops.element('truss', 321, 302, 402, 0.434, IDPT)
ops.element('truss', 322, 303, 403, 0.434, IDPT)
ops.element('elasticBeamColumn', 331, 402, pie2node[-1], 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
ops.element('elasticBeamColumn', 332, pie2node[-1], 403, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
print('element', 'truss', 301, 300, 400, 0.434, IDPT)
print('element', 'truss', 302, 301, 401, 0.434, IDPT)
print('element', 'elasticBeamColumn', 311, 400, pie1node[-1], 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
print('element', 'elasticBeamColumn', 312, pie1node[-1], 401, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
print('element', 'truss', 321, 302, 402, 0.434, IDPT)
print('element', 'truss', 322, 303, 403, 0.434, IDPT)
print('element', 'elasticBeamColumn', 331, 402, pie2node[-1], 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
print('element', 'elasticBeamColumn', 332, pie2node[-1], 403, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
# Define Rigid Beam
ops.element('elasticBeamColumn', 401, 2000, 2 * len(element_length) + 1, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
ops.element('elasticBeamColumn', 402, 2 * len(element_length) + 1, 2 * len(element_length) + 2, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
ops.element('elasticBeamColumn', 403, 2 * len(element_length) + 2, 4000, 1e4, 3.0e10, 9.0e4, geomTransfTag_PDelta)
print('elasticBeamColumn', 401, 2000, 2 * len(element_length) + 1, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
print('elasticBeamColumn', 402, 2 * len(element_length) + 1, 2 * len(element_length) + 2, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
print('elasticBeamColumn', 403, 2 * len(element_length) + 2, 4000, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)

dataDir = 'SFI_MVLEM_DCRP_1'
if not os.path.exists(dataDir):
    os.mkdir(dataDir)
os.chdir(dataDir)


N = 194.77   # in N
IDctrlNode1 = 2 * len(element_length) + 1
IDctrlNode2 = 2 * len(element_length) + 2
# print('IDctrlNode1 =', IDctrlNode1, 'IDctrlNode2 =', IDctrlNode2,)
ControlNodeDof = 1

ops.recorder('Node', '-file', f'MVLEM_Dtop.txt', '-time', '-node', IDctrlNode1, '-dof', 1, 'disp')

ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
ops.load(IDctrlNode1, 0, -N, 0)
ops.load(IDctrlNode2, 0, -N, 0)

# ------------------------------
# Analysis generation
# ------------------------------
# Create the integration scheme, the LoadControl scheme using steps of 0.1
ops.integrator('LoadControl', 0.1)
# Create the system of equation, a sparse solver with partial pivoting
ops.system('BandGen')
# Create the convergence test, the norm of the residual with a tolerance of 1e-5 and a max number of iterations of 100
ops.test('NormDispIncr', 1.0e-2, 100)
# Create the DOF numberer, the reverse Cuthill-McKee algorithm
ops.numberer('RCM')
# Create the constraint handler, the transformation method
ops.constraints('Transformation')
# Create the solution algorithm, a Newton-Raphson algorithm
ops.algorithm('Newton')
# Create the analysis object
ops.analysis('Static')
# Run analysis
ops.analyze(10)
ops.loadConst('-time', 0.0)
ok = ops.analyze(10)
print("Gravity Analysis is completed, return value is", ok)

ops.loadConst(0.0)

ops.timeSeries('Linear', 2)
ops.pattern('Plain', 2, 2)
ops.load(IDctrlNode1, 1, 0, 0)
ops.constraints('Penalty', 1e20, 1e20)
ops.numberer('RCM')
ops.system('BandGen')
ops.test('NormDispIncr', 1e-2, 2000)
ops.algorithm('Newton')
ops.analysis('Static')
displacement = [int(15000 * 0.03937), int(20000 * 0.03937), int(30000 * 0.03937), int(40000 * 0.03937), int(50000 * 0.03937), int(60000 * 0.03937), int(80000 * 0.03937), int(100000 * 0.03937), int(120000 * 0.03937)]
for i in range(0, len(displacement)):
    disp = displacement[i]
    ops.integrator('DisplacementControl', IDctrlNode1, 1, 0.001)
    ops.analyze(disp)
    ops.integrator('DisplacementControl', IDctrlNode1, 1, -0.001)
    ops.analyze(disp)
    ops.integrator('DisplacementControl', IDctrlNode1, 1, -0.001)
    ops.analyze(disp)
    ops.integrator('DisplacementControl', IDctrlNode1, 1, 0.001)
    ops.analyze(disp)







# ---------------------
# Define Axial Load
# ---------------------

N = 194.77   # in kip
IDctrlNode1 = 2 * len(element_length) + 1
IDctrlNode2 = 2 * len(element_length) + 2
# print('IDctrlNode1 =', IDctrlNode1, 'IDctrlNode2 =', IDctrlNode2,)
ControlNodeDof = 1

ops.recorder('Node', '-file', f'MVLEM_Dtop.txt', '-time', '-node', IDctrlNode1, '-dof', 1, 'disp')
# -------------------------------------------------------
# Set parameters for displacement controlled analysis
# -------------------------------------------------------
# vector of displacement-cycle peaks in terms of wall drift ratio
iDmax = np.array([0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.035, 0.04, 0.05])
Dincr = 0.001   # displacement increment for displacement controlled analysis
Ncycle = 1      # Specify the number of cycles at each peak
Tol = 1.0e-2
Lunit = 'inch'

# ------------------------------
# Create a Plain load pattern with a linear TimeSeries
# ------------------------------

ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
ops.load(IDctrlNode1, 0.0, -N, 0.0)
ops.load(IDctrlNode2, 0.0, -N, 0.0)
# ------------------------------
# Analysis generation
# ------------------------------
# Create the integration scheme, the LoadControl scheme using steps of 0.1
ops.integrator('LoadControl', 0.1)
# Create the system of equation, a sparse solver with partial pivoting
ops.system('BandGen')
# Create the convergence test, the norm of the residual with a tolerance of 1e-5 and a max number of iterations of 100
ops.test('NormDispIncr', Tol, 100)
# Create the DOF numberer, the reverse Cuthill-McKee algorithm
ops.numberer('RCM')
# Create the constraint handler, the transformation method
ops.constraints('Transformation')
# Create the solution algorithm, a Newton-Raphson algorithm
ops.algorithm('Newton')
# Create the analysis object
ops.analysis('Static')
# Run analysis
ops.analyze(10)
ops.loadConst('-time', 0.0)
ok = ops.analyze(10)
print("Gravity Analysis is completed, return value is", ok)

# ------------------------------
# Plot the model (opsvis)
# ------------------------------
# opsv.plot_model()
# plt.title('Plot_model after defining elements')
# opsv.plot_loads_2d()
# plt.show()

# ------------------------------
# Plot the model (vfo)
# ------------------------------
# vfo.plot_model()
# plt.show()
print("-----------------------------------------------------------------")
print("<<<<< Model generated and gravity load applied successfully >>>>>")
print("<<<<<<<<<<<<< Start Displacement Controlled Analysis >>>>>>>>>>>>")
# ----------------------------------------------------
# Fact = H  # scale drift ratio by story height for displacement cycles
# Set lateral load pattern with a Linear TimeSeries
Plateral = 1.0
ops.timeSeries('Linear', 2)
ops.pattern('Plain', 200, 2)
ops.load(IDctrlNode1, Plateral, 0.0, 0.0)
load_step = 1
# Static analysis parameters
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('BandGen')
ops.test('NormDispIncr', Tol, 100, 1)
ops.algorithm('Newton')
ops.integrator('DisplacementControl', IDctrlNode1, ControlNodeDof, Dincr)
ops.analysis('Static')

# perform Static Cyclic Displacements Analysis
for Dmax in iDmax:
    # print(Dmax)
    iDstep = NewGeneratePeaks.NewGeneratePeaks(Dmax, Dincr, 'Full', pielen)
    # print(iDstep)

    # print(iDstep)
    for i in range(1, int(Ncycle + 1)):
        zeroD = 0
        D0 = 0
        ok = 1
        for Dstep in iDstep:
            D1 = Dstep
            Dincr = D1 - D0
            ops.integrator('DisplacementControl', IDctrlNode1, ControlNodeDof, Dincr)
            ops.analysis('Static')
            # ------------------------- first analyze command ------------------------
            ok = ops.analyze(1)
            # ------------------------ if convergence failure -------------------------
            if ok != 0:
                if ok != 0:
                    print("Trying Newton with Initial Tangent --")
                    ops.test('NormDispIncr', Tol, 2000, 0)
                    ops.algorithm('Newton', '-initial')
                    ok = ops.analyze(1)
                    ops.test('NormDispIncr', Tol, 100, 0)
                    ops.algorithm('Newton')
                if ok != 0:
                    print("Trying Broyden --")
                    ops.test('NormDispIncr', Tol, 2000, 0)
                    ops.algorithm('Broyden', 500)
                    ok = ops.analyze(1)
                    ops.test('NormDispIncr', Tol, 100, 0)
                    ops.algorithm('Newton')
                if ok != 0:
                    print("PROBLEM Cyclic analysis : CtrlNode ", IDctrlNode1, "CtrlDOF ", ControlNodeDof, "Disp = ",
                          ops.nodeDisp(IDctrlNode1, ControlNodeDof), Lunit)
            D0 = D1
            print("D0 =", D0)
            print("Load Step : ", load_step)
            load_step = load_step + 1
if ok != 0:
     print("PROBLEM Cyclic analysis : CtrlNode ", IDctrlNode1, "CtrlDOF ", ControlNodeDof, "Disp = ",
           ops.nodeDisp(IDctrlNode1, ControlNodeDof), Lunit)
else:
    print("Done Cyclic analysis : CtrlNode ", IDctrlNode1, "CtrlDOF ", ControlNodeDof, "Disp = ",
          ops.nodeDisp(IDctrlNode1, ControlNodeDof), Lunit)
