import math
import os
import time

import h5py
import numpy as np
import openseespy.opensees as ops
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.io import savemat

from Units import *


tw = 400
hw = 2600
lw = 1600
lbe = 200
fc = 40
fyb = 340
fyw = 340
rouYb = 0.003
rouYw = 0.003
rouXb = 0.003
rouXw = 0.003
loadCoeff = 0.1
unblED = 400
anclED = 0
anclPT = 600
numENT = 9
numED = 4
numPT = 2
locED = [-135, -120, 120, 135]
locPT = [-100, 100]


def build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, numENT, numED, numPT, eleH=16, eleL=8, printProgression=True):
    ops.wipe()
    ops.model('Basic', '-ndm', 2, '-ndf', 3)

    # ----------------------------------------------------------------------------------------
    # Set geometry, ops.nodes, boundary conditions
    # ----------------------------------------------------------------------------------------
    wall_thickness = tw  # Wall thickness
    wall_height = hw  # Wall height
    wall_length = lw  # Wall width
    length_be = lbe  # Length of the Boundary Element
    length_web = lweb = lw - (2 * lbe)  # Length of the Web

    # ----------------------------------------------------------------------------------------
    # Discretization of the wall geometry
    # ----------------------------------------------------------------------------------------
    m = eleH
    n = eleL
    eleBE = 2
    eleWeb = eleL - eleBE
    elelweb = lweb / eleWeb

    # ----------------------------------------------------------------------------------------
    # Define Nodes (for MVLEM)
    # ----------------------------------------------------------------------------------------
    for i in range(2, eleH + 2):
        ops.node(i, 0, (i - 1) * (hw / eleH))
        print('Node', i, 0, (i - 1) * (hw / eleH))

    # ---------------------------------------------------------------------------------------
    # Define Control Node and DOF                                                   ||
    # ---------------------------------------------------------------------------------------
    global ControlNode, ControlNodeDof
    ControlNode = eleH + 1  # Control Node (TopNode)
    ControlNodeDof = 1  # Control DOF 1 = X-direction
    print('ControlNode', ControlNode)

    # Pier 1 base
    # [96,  97,  98,  99,  100, 101, 102, 103, 104]
    #   |    |    |    |    |    |    |    |    |
    # [86,  87,  88,  89,  90,  91,  92,  93,  94]

    bas1botnodes = []
    bas1topnodes = []
    top1botnodes = []

    for i in range(0, numENT):
        bas1botnodes.append(int(86 + i))
        bas1topnodes.append(int(86 + numENT + 1 + i))
    # print(bas1botnodes)
    # print(bas1topnodes)

    for i in range(0, numENT):
        ops.node(bas1botnodes[i], -lw / 2 + i * lw / (numENT - 1), 0)
        ops.node(bas1topnodes[i], -lw / 2 + i * lw / (numENT - 1), 0)
        print('node', bas1botnodes[i], -lw / 2 + i * lw / (numENT - 1), 0)
        print('node', bas1topnodes[i], -lw / 2 + i * lw / (numENT - 1), 0)

    pie1node = []
    pie1node.append(bas1topnodes[int((numENT - 1) / 2)])
    for i in range(1, eleH + 1):
        pie1node.append(i + 1)
    print(pie1node)

    # Energy Dissipating Rebars Node
    for j in range(0, numED):
        ops.node(200 + j, locED[j], -anclED)
        ops.node(300 + j, locED[j], unblED)
        print('node', 200 + j, locED[j], -anclED)
        print('node', 300 + j, locED[j], unblED)

    # PT Node
    for j in range(0, numPT):
        ops.node(400 + j, locPT[j], -anclPT)
        ops.node(500 + j, locPT[j], hw)
        print('node', 400 + j, locPT[j], -anclPT)
        print('node', 500 + j, locPT[j], hw)

    # ------------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------------
    # Constraints of base spring bottom nodes of pier1 and pier2
    for i in range(0, len(bas1botnodes)):
        ops.fix(bas1botnodes[i], 1, 1, 1)
        print('fix', bas1botnodes[i], 1, 1, 1)

    # Constraints of base spring top nodes of pier1 and pier2
    for i in range(0, 1):
        ops.fix(bas1topnodes[i], 1, 0, 0)
        print('fix', bas1topnodes[i], 1, 0, 0)

    # Constraints of ED truss bottom nodes of pier1 and pier2
    for j in range(0, numED):
        ops.fix(200 + j, 1, 1, 1)
        print('fix', 200 + j, 1, 1, 1)

    # Constraints of PT truss bottom nodes of pier1 and pier2
    for j in range(0, numPT):
        ops.fix(400 + j, 1, 1, 1)
        print('fix', 400 + j, 1, 1, 1)

    # ---------------------------------------------------------------------------------------
    # Define Steel uni-axial materials
    # ---------------------------------------------------------------------------------------
    sYb = 1
    sYw = 2

    # STEEL misc
    Es = 200 * GPa  # Young's modulus

    # STEEL Y BE (boundary element)
    fyYbp = fyb  # fy - tension
    fyYbn = fyb  # fy - compression
    bybp = 0.01  # strain hardening - tension
    bybn = 0.01  # strain hardening - compression

    # STEEL Y WEB
    fyYwp = fyw  # fy - tension
    fyYwn = fyw  # fy - compression
    bywp = 0.02  # strain hardening - tension
    bywn = 0.02  # strain hardening - compression

    # STEEL X
    fyXp = fyw  # fy - tension
    fyXn = fyw  # fy - compression
    bXp = 0.02  # strain hardening - tension
    bXn = 0.02  # strai n hardening - compression

    # STEEL misc
    Bs = 0.01  # strain-hardening ratio
    R0 = 20.0  # initial value of curvature parameter
    cR1 = 0.925  # control the transition from elastic to plastic branches
    cR2 = 0.0015  # control the transition from elastic to plastic branches

    # Steel ED
    IDED = 3
    fyED = 291
    EED = 200000
    b = 0.01

    # Steel PT
    IDPT = 4
    fyPT = 1860
    EPT = 200000
    a1 = 0
    a2 = 1
    a3 = 0
    a4 = 1
    sigInit = 750

    # SteelMPF model
    ops.uniaxialMaterial('SteelMPF', sYb, fyYbp, fyYbn, Es, bybp, bybn, R0, cR1, cR2)  # Steel Y boundary
    ops.uniaxialMaterial('SteelMPF', sYw, fyYwp, fyYwn, Es, bywp, bywn, R0, cR1, cR2)  # Steel Y web
    ops.uniaxialMaterial('Steel02', IDED, fyED, EED, b, R0, cR1, cR2)  # Steel ED
    ops.uniaxialMaterial('Steel02', IDPT, fyPT, EPT, b, R0, cR1, cR2, a1, a2, a3, a4, sigInit)  # Steel PT

    # ---------------------------------------------------------------------------------------
    # Define "ConcreteCM" uni-axial materials
    # ---------------------------------------------------------------------------------------
    concWeb = 5
    concBE = 6

    # ----- unconfined concrete for WEB
    fc0 = abs(fc) * MPa  # Initial concrete strength
    Ec0 = 8200.0 * (fc0 ** 0.375)  # Initial elastic modulus
    fcU = -fc0 * MPa  # Unconfined concrete strength
    ecU = -(fc0 ** 0.25) / 1150  # Unconfined concrete strain
    EcU = Ec0  # Unconfined elastic modulus
    ftU = 0.35 * (fc0 ** 0.5)  # Unconfined tensile strength
    etU = 2.0 * ftU / EcU  # Unconfined tensile strain
    xpU = 2.0
    xnU = 2.3
    rU = -1.9 + (fc0 / 5.2)  # Shape parameter
    # ----- confined concrete for BE
    fl1 = -1.58 * MPa  # Lower limit of confined concrete strength
    fl2 = -1.87 * MPa  # Upper limit of confined concrete strength
    q = fl1 / fl2
    x = (fl1 + fl2) / (2.0 * fcU)
    A = 6.8886 - (0.6069 + 17.275 * q) * math.exp(-4.989 * q)
    B = (4.5 / (5 / A * (0.9849 - 0.6306 * math.exp(-3.8939 * q)) - 0.1)) - 5.0
    k1 = A * (0.1 + 0.9 / (1 + B * x))
    # Check the strength of transverse reinforcement and set k2 accordingly10.47
    if abs(fyb) <= 413.8 * MPa:  # Normal strength transverse reinforcement (<60ksi)
        k2 = 5.0 * k1
    else:  # High strength transverse reinforcement (>60ksi)
        k2 = 3.0 * k1
    # Confined concrete properties
    fcC = fcU * (1 + k1 * x)
    ecC = ecU * (1 + k2 * x)  # confined concrete strain
    EcC = Ec0
    ftC = ftU
    etC = etU
    xpC = xpU
    xnC = 30.0
    ne = EcC * ecC / fcC
    rC = ne / (ne - 1)

    ru = 7.3049  # shape parameter - compression
    xcrnu = 1.0125  # cracking strain - compression
    rc = 7  # shape parameter - compression
    xcrnc = 1.039  # cracking strain - compression
    et = 0.00008  # strain at peak tensile stress (0.00008)
    rt = 1.2  # shape parameter - tension
    xcrp = 10000  # cracking strain - tension

    # -------------------------- ConcreteCM model --------------------------------------------
    ops.uniaxialMaterial('ConcreteCM', concWeb, fcU, ecU, EcU, rU, xcrnu, ftU, etU, rt, xcrp, '-GapClose', 0)  # Web (unconfined concrete)
    ops.uniaxialMaterial('ConcreteCM', concBE, fcC, ecC, EcC, rC, xcrnc, ftC, etC, rt, xcrp, '-GapClose', 0)  # BE (confined concrete)
    print('--------------------------------------------------------------------------------------------------')
    print('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, et, rt, xcrp, '-GapClose', 0)  # Web (unconfined concrete)
    print('ConcreteCM', concBE, fcC, ecC, EcC, rc, xcrnc, ftC, et, rt, xcrp, '-GapClose', 0)  # BE (confined concrete)

    # ----------------------------Shear spring for MVLEM-------------------------------------
    Ac = lw * tw  # Concrete Wall Area
    Gc = Ec0 / (2 * (1 + 0.2))  # Shear Modulus G = E / 2 * (1 + v)
    Kshear = Ac * Gc * (5 / 6)  # Shear stiffness k * A * G ---> k=5/6

    # Shear Model for Section Aggregator to assign for MVLEM element shear spring
    ops.uniaxialMaterial('Elastic', 7, Kshear)

    # Define ENT Material
    IDENT = 9
    EENT = 2.5 * 32500.0 * lw * tw / (hw * numENT)
    ops.uniaxialMaterial('ENT', IDENT, EENT)

    # ------------------------------
    #  Define SFI_MVLEM elements
    # ------------------------------
    MVLEM_thick = [tw] * n
    MVLEM_width = [lbe if i in (0, n - 1) else elelweb for i in range(n)]
    MVLEM_rho = [rouYb if i in (0, n - 1) else rouYw for i in range(n)]
    MVLEM_matConcrete = [concBE if i in (0, n - 1) else concWeb for i in range(n)]
    MVLEM_matSteel = [sYb if i in (0, n - 1) else sYw for i in range(n)]

    for i in range(eleH):
        # ------------------ MVLEM ----------------------------------------------
        ops.element('MVLEM', i + 101, 0.0, pie1node[i], pie1node[i + 1], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', 7)
        print('element', 'MVLEM', i + 101, pie1node[i], pie1node[i + 1], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', 7)

    # Define Zero-Length Element
    for i in range(0, numENT):
        ops.element('zeroLength', i + 601, bas1botnodes[i], bas1topnodes[i], '-mat', IDENT, '-dir', 2)
        print('zeroLength', i + 601, bas1botnodes[i], bas1topnodes[i], '-mat', IDENT, '-dir', 2)

    geomTransfTag_PDelta = 1
    ops.geomTransf('PDelta', geomTransfTag_PDelta)
    for i in range(0, numENT - 1):
        ops.element('elasticBeamColumn', i + 1601, bas1topnodes[i], bas1topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
        print('elasticBeamColumn', i + 1601, bas1topnodes[i], bas1topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)

    # Define ED elements
    EDas = 314.11
    for i in range(0, numED):

    for i in range(num_elements):
        ops.element('truss', 203 + i, 202 + i, 302 + i, EDas[i], IDED[i])

        ops.element('truss', 201, 200, 300, EDas, IDED)
        ops.element('truss', 202, 201, 301, EDas, IDED)
        ops.element('truss', 203, 202, 302, EDas, IDED)
        ops.element('truss', 204, 203, 303, EDas, IDED)
    ops.element('elasticBeamColumn', 211, 300, 301, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 212, 301, 2, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 213, 2, 302, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 214, 302, 303, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)

    # print('element', 'truss', 201, 100, 200, 314.11 * 2, IDED)
    # print('element', 'truss', 202, 101, 201, 314.11 * 2, IDED)
    # print('element', 'truss', 203, 102, 202, 353.37 * 1, IDED)
    # print('element', 'truss', 204, 103, 203, 353.37 * 1, IDED)
    # print('element', 'elasticBeamColumn', 211, 200, 201, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    # print('element', 'elasticBeamColumn', 212, 201, 2, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    # print('element', 'elasticBeamColumn', 213, 2, 202, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    # print('element', 'elasticBeamColumn', 214, 202, 203, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)

    # Define PT elements
    PTas = 280
    ops.element('truss', 301, 400, 500, PTas, IDPT)
    ops.element('truss', 302, 401, 501, PTas, IDPT)
    ops.element('elasticBeamColumn', 311, 500, pie1node[-1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 312, pie1node[-1], 501, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)

    # print('element', 'truss', 301, 300, 400, 280, IDPT)
    # print('element', 'truss', 302, 301, 401, 280, IDPT)
    # print('element', 'elasticBeamColumn', 311, 400, pie1node[-1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    # print('element', 'elasticBeamColumn', 312, pie1node[-1], 401, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)

    # ---------------------------------------------------------------------------------------
    # Define Axial Load on Top Node
    # ---------------------------------------------------------------------------------------
    global Aload  # axial force in N according to ACI318-19 (not considering the reinforced steel at this point for simplicity)
    Aload = 0.85 * abs(fc) * tw * lw * loadCoeff
    # print('Axial load fc(kN) = ', Aload / 1000)


def run_gravity(steps=10, printProgression=True):
    if printProgression:
        print("RUNNING GRAVITY ANALYSIS")
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(ControlNode, 0, -Aload, 0)
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1e-6, 100, 0)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 1 / steps)
    ops.analysis('Static')
    ops.analyze(steps)
    ops.loadConst('-time', 0.0)  # hold gravity constant and restart time for further analysis
    if printProgression:
        print("GRAVITY ANALYSIS DONE!")


def cyclic(displacement):
    dataDir = 'DCRP_1_cyclic'
    # os.chdir('..')
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
    os.chdir(dataDir)

    ops.recorder('Node', '-file', f'MVLEM_Dtop.txt', '-time', '-node', ControlNode, '-dof', 1, 'disp')
    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    ops.load(ControlNode, 1000, 0, 0)
    ops.constraints('Penalty', 1e20, 1e20)
    ops.numberer('RCM')
    ops.system('BandGen')
    ops.test('NormDispIncr', 1e-3, 100)
    ops.algorithm('KrylovNewton')
    ops.analysis('Static')

    RESULTS = defaultdict(lambda: [])

    # int(50), int(100), int(150), int(200), int(300), int(400), int(500), int(600), int(800), int(1000)

    for k in range(0, len(displacement)):
        disp = displacement[k]
        ops.integrator('DisplacementControl', ControlNode, 1, 0.1)
        for i in range(0, disp):
            ops.analyze(1)
            # save_result(RESULTS, numENT, element_length)
            f = ops.nodeDisp(ControlNode, 1)
            p = i / disp * 25
            print("1Cycle", k + 1, "/", len(displacement), "Disp = ", '{:.2f}'.format(f), "mm", "------- Processing", '{:.2f}'.format(p), "%-------")
        ops.integrator('DisplacementControl', ControlNode, 1, -0.1)
        for i in range(0, disp):
            ops.analyze(1)
            # save_result(RESULTS, numENT, element_length)
            f = ops.nodeDisp(ControlNode, 1)
            p = i / disp * 25 + 25
            print("1Cycle", k + 1, "/", len(displacement), "Disp = ", '{:.2f}'.format(f), "mm", "------- Processing", '{:.2f}'.format(p), "%-------")
        ops.integrator('DisplacementControl', ControlNode, 1, -0.1)
        for i in range(0, disp):
            ops.analyze(1)
            # save_result(RESULTS, numENT, element_length)
            f = ops.nodeDisp(ControlNode, 1)
            p = i / disp * 25 + 50
            print("1Cycle", k + 1, "/", len(displacement), "Disp = ", '{:.2f}'.format(f), "mm", "------- Processing", '{:.2f}'.format(p), "%-------")
        ops.integrator('DisplacementControl', ControlNode, 1, 0.1)
        for i in range(0, disp):
            ops.analyze(1)
            # save_result(RESULTS, numENT, element_length)
            f = ops.nodeDisp(ControlNode, 1)
            p = i / disp * 25 + 75
            print("1Cycle", k + 1, "/", len(displacement), "Disp = ", '{:.2f}'.format(f), "mm", "------- Processing", '{:.2f}'.format(p), "%-------")

    print("Done Cyclic analysis")

    #print(RESULTS.keys())
    for key, value in RESULTS.items():
        RESULTS[key] = np.array(value)

    file_path = "Result.hdf5"
    with h5py.File(file_path, "w") as f:
        for key, value in RESULTS.items():
            f.create_dataset(key, data=value)

    file_path2 = "Result.mat"
    savemat(file_path2, RESULTS)
    print("Done Saving Data")
    os.chdir('..')


build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, numENT, numED, numPT, eleH=16, eleL=8, printProgression=True)
run_gravity()
cyclic(displacement=[int(50), int(100), int(150), int(200), int(300), int(400), int(500), int(600), int(800), int(1000)])


# Run_DCRP1_cyclic(displacement = [int(50), int(100), int(150), int(200), int(300), int(400), int(500), int(600), int(800), int(1000)])
#


def run_cyclic(DisplacementStep, plotResults=True, printProgression=True, recordData=False):
    if printProgression:
        tic = time.time()
        print("RUNNING CYCLIC ANALYSIS")
    if recordData:
        print("RECORDING SHEAR LOAD VS DISPLACEMENT DATA")
        ops.recorder('Node', '-file', 'RunTimeNodalResults/Cyclic_Reaction.out', '-closeOnWrite', '-node', 1, '-dof', 1, 'reaction')
        ops.recorder('Node', '-file', 'RunTimeNodalResults/Cyclic_Displacement.out', '-closeOnWrite', '-node', ControlNode, '-dof', 1, 'disp')
    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    ops.load(ControlNode, *[1.0, 0.0, 0.0])  # Apply lateral load based on first mode shape in x direction (EC8-1)
    ops.constraints('Penalty')  # Transformation 'Penalty', 1e20, 1e20
    ops.numberer('Plain')
    ops.system("FullGeneral")
    ops.test('EnergyIncr', 1e-8, 1000, 0)
    ops.algorithm('Newton')

    # Define analysis parameters
    Nsteps = len(DisplacementStep)
    finishedSteps = 0
    dispData = np.zeros(Nsteps + 1)
    baseShearData = np.zeros(Nsteps + 1)

    # Perform cyclic analysis
    D0 = 0.0
    for j in range(Nsteps):
        D1 = DisplacementStep[j]
        Dincr = D1 - D0
        if printProgression:
            print(f'Step {j} -------->', f'Dincr = ', Dincr)
        # ------------------------- first analyze command ---------------------------------------------
        ops.integrator("DisplacementControl", ControlNode, ControlNodeDof, Dincr)
        ops.analysis('Static')
        ok = ops.analyze(1)
        # ------------------------ If not converged -------------------------
        # if ok != 0:
        #     ok = analysisLoopDisp(ok, j, Dincr, ControlNode, ControlNodeDof)
        # if ok != 0:
        #     print("Problem running Cyclic analysis for the model : Ending analysis ")
        # break
        # else:
        D0 = D1  # move to next step

        finishedSteps = j + 1
        disp = ops.nodeDisp(ControlNode, 1)
        baseShear = -ops.getLoadFactor(2) / 1000  # Convert to from N to kN
        dispData[j + 1] = disp
        baseShearData[j + 1] = baseShear

        if printProgression:
            print(f'\033[92m InputDisplacement {j} = {DisplacementStep[j]}\033[0m')
            print(f'\033[91mOutputDisplacement {j} = {dispData[j + 1]}\033[0m')

    # if printProgression:
    #     toc = time.time()
    #     print('CYCLIC ANALYSIS DONE IN {:1.2f} seconds'.format(toc - tic))

    # if plotResults:
    #     force_data = np.loadtxt('RunTimeNodalResults/Cyclic_Horizontal_Reaction.out')
    #     displacement_data = np.loadtxt('RunTimeNodalResults/Cyclic_Horizontal_Displacement.out')
    #     # Plot Force vs. Displacement
    #     plt.figure(figsize=(7, 6), dpi=100)
    #     # plt.plot(displacement_data, -force_data, color='blue', linewidth=1.2, label='Numerical test')
    #     plt.plot(dispData, -baseShearData, color="red", linestyle="-", linewidth=1.2, label='Output Displacement vs Shear Load')
    #     # plt.plot([row[0] for row in dataCyc[:finishedSteps]], [-row[1] for row in dataCyc[:finishedSteps]], color="black", linestyle="-", linewidth=1.2, label='Output Displacement vs Shear Load')
    #     # plt.plot(DisplacementStep, [-row[1] for row in dataCyc[:finishedSteps]], color="blue", linestyle="-", linewidth=1.2, label='Input Displacement vs Shear Load')
    #     plt.axhline(0, color='black', linewidth=0.4)
    #     plt.axvline(0, color='black', linewidth=0.4)
    #     plt.grid(linestyle='dotted')
    #     font_settings = {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14}
    #     plt.xlabel('Displacement (mm)', fontdict=font_settings)
    #     plt.ylabel('Base Shear (kN)', fontdict=font_settings)
    #     plt.yticks(fontname='Cambria', fontsize=14)
    #     plt.xticks(fontname='Cambria', fontsize=14)
    #     plt.title(f'Specimen', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    #     plt.tight_layout()
    #     plt.legend()
    #     plt.show()

    return [dispData[0:finishedSteps], -baseShearData[0:finishedSteps]]


def run_pushover(MaxDisp=75, dispIncr=1, plotResults=True, printProgression=True, recordData=False):
    if printProgression:
        tic = time.time()
        print("RUNNING PUSHOVER ANALYSIS")

    if recordData:
        ops.recorder('Node', '-file', 'RunTimeNodalResults/Pushover_Reaction.out', '-node', 1, '-dof', ControlNodeDof, 'reaction')
        ops.recorder('Node', '-file', 'RunTimeNodalResults/Pushover_Displacement.out', '-node', ControlNode, '-dof', ControlNodeDof, 'disp')

    ops.timeSeries('Linear', 3)  # create TimeSeries for gravity analysis
    ops.pattern('Plain', 3, 3)
    ops.load(ControlNode, *[1.0, 0.0, 0.0])  # Apply a unit reference load in DOF=1 (nd    FX  FY  MZ)

    NstepsPush = round(MaxDisp / dispIncr)

    if printProgression:
        print("Starting pushover analysis...")
        print("   total steps: ", NstepsPush)
    ops.constraints('Transformation')
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test('NormDispIncr', 1e-8, 1000)
    ops.algorithm('KrylovNewton')
    ops.analysis("Static")

    maxUnconvergedSteps = 1
    unconvergeSteps = 0
    finishedSteps = 0
    dataPush = np.zeros((NstepsPush + 1, 2))
    dispImpo = np.zeros(NstepsPush + 1)

    # Perform pushover analysis
    for j in range(NstepsPush):
        if unconvergeSteps > maxUnconvergedSteps:
            break
        ops.integrator("DisplacementControl", ControlNode, ControlNodeDof, dispIncr)  # Target node is ControlNode and dof is 1
        ok = ops.analyze(1)
        if ok != 0:
            # ------------------------ If not converged, reduce the increment -------------------------
            unconvergeSteps += 1
            Dts = 20  # Try 50x smaller increments
            smallDincr = dispIncr / Dts
            for k in range(1, Dts):
                if printProgression:
                    print(f'Small Step {k} -------->', f'smallDincr = ', smallDincr)
                ops.integrator("DisplacementControl", ControlNode, ControlNodeDof, smallDincr)
                ok = ops.analyze(1)
            # ------------------------ If not converged --------------------------------------------
            if ok != 0:
                if printProgression:
                    print("Problem running Pushover analysis for the model : Ending analysis ")

        dispImpo += dispIncr
        finishedSteps = j + 1
        disp = ops.nodeDisp(ControlNode, ControlNodeDof)
        baseShear = -ops.getLoadFactor(3) / 1000  # Convert to from N to kN
        dataPush[j + 1, 0] = disp
        dataPush[j + 1, 1] = baseShear

        if printProgression:
            print("step", j + 1, "/", NstepsPush, "   ", "Impos disp = ", round(dispImpo[j], 2), "---->  Real disp = ", str(round(disp, 2)), "---->  dispIncr = ", dispIncr)

    if printProgression:
        toc = time.time()
        print('PUSHOVER ANALYSIS DONE IN {:1.2f} seconds'.format(toc - tic))

    if plotResults:
        plt.rcParams.update({'font.size': 14, "font.family": ["Times New Roman", "Cambria"]})
        plt.figure(figsize=(6, 4), dpi=100)
        plt.plot(dataPush[0:finishedSteps, 0], -dataPush[0:finishedSteps, 1], color="red", linewidth=1.2, linestyle="-", label='Pushover Analysis')
        plt.axhline(0, color='black', linewidth=0.4)
        plt.axvline(0, color='black', linewidth=0.4)
        plt.grid(linestyle='dotted')
        plt.xlabel('Displacement (mm)')
        plt.ylabel('Base Shear (N)')
        plt.tight_layout()
        plt.legend()
        plt.show()

    return [dataPush[0:finishedSteps, 0], -dataPush[0:finishedSteps, 1]]


def run_cyclic2(DisplacementStep, printProgression=True):
    if printProgression:
        tic = time.time()
        print("RUNNING CYCLIC ANALYSIS")
    # define parameters for adaptive time-step
    max_factor = 0.12  # 1.0 -> don't make it larger than initial time step
    min_factor = 1e-06  # at most initial/1e6
    max_factor_increment = 1.5  # define how fast the factor can increase
    min_factor_increment = 1e-06  # define how fast the factor can decrease
    max_iter = 2500
    desired_iter = int(max_iter / 2)  # should be higher than the desired number of iterations

    # -------------CYCLIC-----------------
    ops.timeSeries('Linear', 2, '-factor', 1.0)
    ops.pattern('Plain', 2, 2)
    RefLoad = 1000e3
    ops.load(ControlNode, *[RefLoad, 0.0, 0.0])
    ops.constraints('Transformation')  # Transformation 'Penalty', 1e20, 1e20
    ops.numberer('RCM')
    ops.system("ProfileSPD")
    ops.test('NormDispIncr', 1e-6, desired_iter, 0)
    ops.algorithm('KrylovNewton')  # KrylovNewton
    ops.analysis("Static")

    Nsteps = len(DisplacementStep)
    finishedSteps = 0
    # dispData = np.zeros(Nsteps + 1)
    ShearData = np.zeros(Nsteps + 1)
    D0 = 0.0
    for j in range(Nsteps):
        D1 = DisplacementStep[j]
        Dincr = D1 - D0

        # start with 1 step per Dincr
        n_sub_steps = 1
        # compute the actual displacement increment
        dU = Dincr / n_sub_steps
        dU_tolerance = abs(dU) * 1.0e-8
        factor = 1.0
        old_factor = factor
        dU_cumulative = 0.0
        increment_done = False
        while True:
            # are we done with this cycle?
            if abs(dU_cumulative - Dincr) <= dU_tolerance:
                if printProgression:
                    print("Target displacement has been reached. Current Dincr = {:.3g}".format(dU_cumulative))
                increment_done = True
                break
            # adapt the current displacement increment
            dU_adapt = dU * factor
            if abs(dU_cumulative + dU_adapt) > (abs(Dincr) - dU_tolerance):
                dU_adapt = Dincr - dU_cumulative
            # update integrator
            ops.integrator("DisplacementControl", ControlNode, 1, dU_adapt)
            ok = ops.analyze(1)
            # adapt if necessary
            if ok == 0:
                num_iter = ops.testIter()
                norms = ops.testNorms()
                error_norm = norms[num_iter - 1] if num_iter > 0 else 0.0
                # update adaptive factor (increase)
                factor_increment = min(max_factor_increment, desired_iter / num_iter)
                factor *= factor_increment
                if factor > max_factor:
                    factor = max_factor
                if factor > old_factor:
                    if printProgression:
                        print("Increasing increment factor due to faster convergence. Factor = {:.3g}".format(factor))
                old_factor = factor
                dU_cumulative += dU_adapt
            else:
                num_iter = max_iter
                factor_increment = max(min_factor_increment, desired_iter / num_iter)
                factor *= factor_increment
                if printProgression:
                    print("Reducing increment factor due to non convergence. Factor = {:.3g}".format(factor))
                if factor < min_factor:
                    if printProgression:
                        print("ERROR: current factor is less then the minimum allowed ({:.3g} < {:.3g})".format(factor, min_factor))
                        print("ERROR: the analysis did not converge")
                    break
        if not increment_done:
            break
        else:
            D0 = D1  # move to next step

        finishedSteps = j + 1
        # disp = ops.nodeDisp(ControlNode, 1)
        baseShear = -ops.getLoadFactor(2) / 1000 * RefLoad  # Convert to from N to kN
        # dispData[j + 1] = disp
        ShearData[j + 1] = baseShear
    # return [dispData[0:finishedSteps], -ShearData[0:finishedSteps]]
    return -ShearData[0:finishedSteps]
