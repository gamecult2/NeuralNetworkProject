import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from Units import *

def reset_analysis():
    """
    Resets the analysis by setting time to 0,
    removing the recorders and wiping the analysis.
    """
    ops.setTime(0.0)    ##  Set the time in the Domain to zero
    ops.loadConst()    # Set the loads constant in the domain
    ops.remove('recorders')    # Remove all recorder objects.
    ops.wipeAnalysis()    # destroy all components of the Analysis object
    ops.wipe()

def RebarArea(RebarDiametermm):
    a = 3.1416 * (RebarDiametermm / 2) ** 2  # compute area
    return a

def generate_cyclic_load(duration=6.0, sampling_rate=50, max_displacement=75):
    # Generate a constant time array
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Calculate the displacement slope to achieve the desired max_displacement
    displacement_slope = (max_displacement / 2) / (duration / 2)

    # Generate the cyclic load with displacement varying over time
    cyclic_load = (displacement_slope * t) * np.sin(2 * np.pi * t)

    return cyclic_load

def generate_cyclic_loading_history(duration_per_amplitude, max_displacement):
    # Set the number of amplitudes to 12
    num_amplitudes = 10
    num_per_amplitudes = 2

    # Create a time array for one cycle
    t_one_cycle = np.linspace(0, duration_per_amplitude, num=100)

    # Initialize an empty array to store the loading history
    loading_history = []

    # Generate the loading history with exponentially increasing displacements
    for i in range(num_amplitudes):
        # Calculate displacement exponentially
        displacement = max_displacement * (1.4 ** i)

        # Generate two cycles at each displacement
        for _ in range(num_per_amplitudes):
            loading_history = np.concatenate((loading_history, displacement * np.sin(2 * np.pi * t_one_cycle)))

    # Create a time array for the entire loading history with two decimal places
    t_total = np.round(np.linspace(0, duration_per_amplitude * num_amplitudes * num_per_amplitudes, num=len(loading_history)), 2)

    return t_total, loading_history

def build_model(tw, hw, lw, lbe, fc, fy, rouYb, rouYw, loadcoef, eleH=16, eleL=8):
    ops.wipe()  # Clear opensees model
    ops.model('basic', '-ndm', 2, '-ndf', 3)  # Model of 2 dimensions, 3 dof per node

    # ----------------------------------------------------------------------------------------
    # Set geometry, ops.nodes, boundary conditions
    # ----------------------------------------------------------------------------------------
    # Wall Geometry
    wall_thickness = tw  # Wall thickness
    wall_height = hw  # Wall height
    wall_length = lw  # Wall width
    length_be = lbe  # Length of the Boundary Element
    length_web = lweb = lw - (2 * lbe)  # Length of the Web

    # ----------------------------------------------------------------------------------------
    # Discretization of the wall geometry
    # ----------------------------------------------------------------------------------------
    # number of elements to discretize each boundary element
    m = eleH
    n = eleL
    eleBE = 2
    eleWeb = eleL - eleBE
    elelweb = lweb / eleWeb

    # Loop through the list of node values
    for i in range(1, eleH + 2):
        ops.node(i, 0, (i - 1) * (hw / eleH))

    # Boundary conditions
    ops.fix(1, 1, 1, 1)  # Fixed condition at node 1

    # Set Control Node and DOF
    global IDctrlNode, IDctrlDOF
    IDctrlNode = eleH + 1  # Control Node (TopNode)
    IDctrlDOF = 1  # Control DOF 1 = X-direction

    # ---------------------------------------------------------------------------------------
    # Define uni-axial materials
    # ---------------------------------------------------------------------------------------
    # STEEL Y boundary element
    fyYbp = fy  # fy - tension
    fyYbn = fy  # fy - compression
    bybp = 0.0185  # strain hardening - tension
    bybn = 0.02  # strain hardening - compression

    # STEEL Y web
    fyYwp = fy  # fy - tension
    fyYwn = fy  # fy - compression
    bywp = 0.035  # strain hardening - tension
    bywn = 0.02  # strain hardening - compression

    # STEEL misc
    Es = 200 * GPa  # Young's modulus
    R0 = 20.0  # initial value of curvature parameter
    a1 = 0.925  # curvature degradation parameter
    a2 = 0.0015  # curvature degradation parameter
    Bs = 0.01  # strain-hardening ratio
    cR1 = 0.925  # control the transition from elastic to plastic branches
    cR2 = 0.15  # control the transition from elastic to plastic branches

    # Build steel materials
    ops.uniaxialMaterial('SteelMPF', 1, fyYbp, fyYbn, Es, bybp, bybn, R0, a1, a2)  # steel Y boundary
    ops.uniaxialMaterial('SteelMPF', 2, fyYwp, fyYwn, Es, bywp, bywn, R0, a1, a2)  # steel Y web
    ops.uniaxialMaterial('MinMax', 6, 1, '-min', -0.06, '-max', 0.06)
    ops.uniaxialMaterial('MinMax', 7, 2, '-min', -0.06, '-max', 0.06)

    # STEEL Reinforcing steel
    # ops.uniaxialMaterial('Steel02', 1, fyYbp, Es, Bs, R0, cR1, cR2)  # steel Y boundary
    # ops.uniaxialMaterial('Steel02', 2, fyYwp, Es, Bs, R0, cR1, cR2)  # steel Y web
    # ops.uniaxialMaterial('MinMax', 6, 1, '-min', -0.06, '-max', 0.06)
    # ops.uniaxialMaterial('MinMax', 7, 2, '-min', -0.06, '-max', 0.06)
    # '''

    # CONCRETE misc -------------------------------------------------------------------------
    # unconfined
    fpc = fc                 # Concrete Compressive Strength, MPa   (+Tension, -Compression)
    Ec = 37 * GPa            # Young's modulus
    ec0 = 2 * (fpc / Ec)     # strain at maximum compressive stress (-0.0021)
    ru = 7                   # shape parameter - compression
    xcrnu = 1.039            # cracking strain - compression

    # confined
    # Ecc = 30 * GPa           # Young's modulus
    # fpcc = fc * 1.2          # Concrete Compressive Strength, MPa   (+Tension, -Compression)
    # ec0c = 2 * (fpcc / Ecc)  # strain at maximum compressive stress (-0.0033)
    # rc = 7.3049              # shape parameter - compression
    # xcrnc = 1.0125           # cracking strain - compression
    # ft = 2.03 * MPa          # peak tensile stress
    # et = 0.00008             # strain at peak tensile stress
    # rt = 1.2                 # shape parameter - tension
    # xcrp = 10000             # cracking strain - tension

    # Build concrete materials
    # ops.uniaxialMaterial('ConcreteCM', 3, fpc, ec0, Ec, ru, xcrnu, ft, et, rt, xcrp)     # unconfined concrete
    # ops.uniaxialMaterial('ConcreteCM', 4, fpcc, ec0c, Ecc, rc, xcrnc, ft, et, rt, xcrp)  # confined concrete
    # print('ConcreteCM', 3, fpc, ec0, Ec, ru, xcrnu, ft, et, rt, xcrp)     # unconfined concrete
    # print('ConcreteCM', 4, fpcc, ec0c, Ecc, rc, xcrnc, ft, et, rt, xcrp)  # confined concrete

    # CONCRETE07 misc -------------------------------------------------------------------------
    # Material properties
    fc0 = abs(fc) * MPa  # Initial concrete strength
    Ec0 = 8200.0 * (fc0 ** 0.375) * MPa  # Initial elastic modulus
    fcU = -fc0 * MPa  # Unconfined concrete strength
    ecU = -pow(fc0, 0.25) / 1152.7  # Unconfined concrete strain
    EcU = Ec0  # Unconfined elastic modulus
    ftU = 1.0e-2 * 0.5 * (fc0 ** 0.5) * MPa  # Unconfined tensile strength
    etU = 2.0 * ftU / EcU  # Unconfined tensile strain
    xpU = 2.0
    xnU = 2.3
    rU = fc0 / 5.2 - 1.9

    fl1 = -1.58 * MPa  # Lower limit of confined concrete strength
    fl2 = -1.87 * MPa  # Upper limit of confined concrete strength
    q = fl1 / fl2
    x = (fl1 + fl2) / (2.0 * fcU)
    A = 6.8886 - (0.6069 + 17.275 * q) * math.exp(-4.989 * q)
    B = 4.5 / (5 / (A * (0.9849 - 0.6306 * math.exp(-3.8939 * q)) - 0.1) - 5.0)
    k1 = A * (0.1 + 0.9 / (1 + B * x))
    fyh = fy * MPa  # Yield strength of transverse reinforcement
    # Check the strength of transverse reinforcement and set k2 accordingly
    if abs(fyh) <= 413.8 * MPa:  # Normal strength transverse reinforcement (<60ksi)
        k2 = 5.0 * k1
    else:  # High strength transverse reinforcement (>60ksi)
        k2 = 3.0 * k1
    # Confined concrete properties
    fcC = fcU * (1 + k1 * x)
    ecC = ecU * (1 + k2 * x)
    EcC = Ec0
    ftC = ftU
    etC = etU
    xpC = xpU
    xnC = 30.0
    ne = EcC * ecC / fcC
    rC = ne / (ne - 1)

    ops.uniaxialMaterial('Concrete07', 3, fcU, ecU, EcU, ftU, etU, xpU, xnU, rU)
    ops.uniaxialMaterial('Concrete07', 4, fcC, ecC, EcC, ftC, etC, xpC, xnC, rC)
    # print('Concrete07', 3, fcU, ecU, EcU, ftU, etU, xpU, xnU, rU)
    # print('Concrete07', 4, fcC, ecC, EcC, ftC, etC, xpC, xnC, rC)

    '''
    # CONCRETE ---------------------------------------------------------------
    # fc = fc     # Concrete Compressive Strength, MPa   (+Tension, -Compression)
    Ec = 37 * GPa  # Concrete Elastic Modulus
    # unconfined concrete
    fc1U = fc  # unconfined concrete (todeschini parabolic model), maximum stress
    eps1U = -0.0021  # strain at maximum strength of unconfined concrete
    fc2U = 0.2 * fc1U  # ultimate stress
    eps2U = -0.01  # strain at ultimate stress
    lam = 0.1  # ratio between unloading slope at $eps2 and initial slope $Ec
    # confined concrete
    fc1C = fc * 1.2  # confined concrete (mander model), maximum stress
    eps1C = 2 * fc1C / Ec  # strain at maximum stress (-0.0033)
    fc2C = 0.2 * fc1C  # ultimate stress
    eps2C = 5 * eps1C  # strain at ultimate stress
    # tensile-strength properties
    ftC = 2.03 * MPa  # tensile strength +tension
    ftU = 2.03 * MPa  # tensile strength +tension
    Ets = ftU / 0.002  # tension softening stiffness

    ops.uniaxialMaterial('Concrete02', 3, fc1U, eps1U, fc2U, eps2U, lam, ftU, Ets)  # COVER CONCRETE  (unconfined)
    ops.uniaxialMaterial('Concrete02', 4, fc1C, eps1C, fc2C, eps2C, lam, ftC, Ets)  # CORE CONCRETE  (confined)
    # print('Concrete02', 3, fc1U, eps1U, fc2U, eps2U, lam, ftU, Ets)  # COVER CONCRETE  (unconfined)
    # print('Concrete02', 4, fc1C, eps1C, fc2C, eps2C, lam, ftC, Ets)  # CORE CONCRETE  (confined)
    '''  # Other Concrete02 Model

    # Shear spring -----------------------------------------------------------
    Ac = lw * tw  # Concrete Wall Area
    Gc = Ec / (2 * (1 + 0.2))  # Shear Modulus G = E / 2 * (1 + v)
    Kshear = Ac * Gc * (5 / 6)  # Shear stiffness k * A * G ---> k=5/6

    # Build shear material to assign for MVLEM element shear spring
    ops.uniaxialMaterial('Elastic', 5, Kshear)  # Shear Model for Section Aggregator

    # axial force in N according to ACI318-19 (not considering the reinforced steel at this point for simplicity)
    global Aload
    # Aload = 0.07 * abs(fc) * tw * lw * loadcoef
    Aload = 0.85 * abs(fc) * tw * lw * loadcoef
    # print('Axial load (kN) = ', Aload / 1000)
    # ------------------------------------------------------------------------
    #  Calculate the parameters for 'MVLEM' elements
    # ------------------------------------------------------------------------
    # Calculate 'MVLEM' elements reinforcement Ratios (RebarArea / ConcreteArea)
    # rouYb = (RebarArea(bereinfDiam) * bereinfNum) / (lbe * tw)  # Y boundary
    # rouYw = (RebarArea(webreinfDiam) * webreinfNum) / (lweb * tw)  # Y web
    # rouYw = rouYw/eleWeb
    # rouYb = 0.030
    # rouYw = 0.0020
    # print('rouYb = ', rouYb)
    # print('rouYw = ', rouYw)

    # ------------------------------------------------------------------------
    #  Define 'MVLEM' elements
    # ------------------------------------------------------------------------
    # Set 'MVLEM' parameters thick, width, rho, matConcrete, matSteel
    MVLEM_thick = [tw] * n
    MVLEM_width = [lbe if i in (0, n - 1) else elelweb for i in range(n)]
    MVLEM_rho = [rouYb if i in (0, n - 1) else rouYw for i in range(n)]
    MVLEM_matConcrete = [4 if i in (0, n - 1) else 3 for i in range(n)]
    MVLEM_matSteel = [1 if i in (0, n - 1) else 2 for i in range(n)]

    for i in range(eleH):
        ops.element('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', 5)
        # print('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', 5)

def run_analysis(DisplacementStep, plotPushOverResults=True, printProgression=True):

    if printProgression:
        print("RUNNING GRAVITY ANALYSIS")

    steps = 10
    # ops.recorder('Node', '-file', 'RunTimeNodalResults/Disp.txt', '-closeOnWrite', '-time', '-node', IDctrlNode, '-dof', 2, 'disp')
    # ops.recorder('Node', '-file', 'RunTimeNodalResults/Gravity_Reactions.out', '-time', '-node', *[1], '-dof', *[1, 2, 3], 'reaction')

    ops.timeSeries('Linear', 1)  # create TimeSeries for gravity analysis
    ops.pattern('Plain', 1, 1)
    ops.load(IDctrlNode, *[0.0, -Aload, 0.0])  # apply vertical load

    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-6, 100, 0)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 1 / steps)
    ops.analysis('Static')
    ops.analyze(steps)

    if printProgression:
        print("GRAVITY ANALYSIS DONE!")

    # Keep the gravity loads for further analysis
    ops.loadConst('-time', 0.0)  # hold gravity constant and restart time

    # START OF CYCLIC LOADING ANALYSIS
    if printProgression:
        print("RUNNING CYCLIC ANALYSIS")

    tic = time.time()
    ops.recorder('Node', '-file', 'RunTimeNodalResults/Cyclic_Horizontal_Reaction.out', '-closeOnWrite', '-node', 1, '-dof', 1, 'reaction')
    ops.recorder('Node', '-file', 'RunTimeNodalResults/Cyclic_Horizontal_Displacement.out', '-closeOnWrite', '-node', IDctrlNode, '-dof', 1, 'disp')

    # Apply lateral load based on first mode shape in x direction (EC8-1)
    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    ops.load(IDctrlNode, *[1.0, 0.0, 0.0])  # apply vertical load

    ops.constraints('Penalty', 1e20, 1e20)
    ops.numberer('RCM')
    ops.system("BandGeneral")
    ops.test('NormDispIncr', 1e-9, 100, 0)
    ops.algorithm('KrylovNewton')
    ops.analysis('Static')

    # D0 = 0.0
    # for Dstep in DisplacementStep:
    #     D1 = Dstep
    #     Dincr = D1 - D0
    #     ops.integrator("DisplacementControl", IDctrlNode, 1, Dincr)
    #     ok = ops.analyze(1)
    #     # ----------------------------------------------if convergence failure-------------------------
    #     D0 = D1  # move to next step
    #     # end Dstep
    #     if ok != 0:
    #         print("Analysis failed at {} step.".format(Dstep))
    #     else:
    #         print("Analysis completed successfully.") # DisplacementStep

    maxUnconvergedSteps = 10
    unconvergeSteps = 0
    Nsteps = len(DisplacementStep)
    finishedSteps = 0
    dataPush = np.zeros((Nsteps + 1, 2))

    # Perform pushover analysis
    D0 = 0.0
    for j in range(Nsteps):
        D1 = DisplacementStep[j]
        Dincr = D1 - D0
        if printProgression:
            print(f'DisplacementStep {j} = ', DisplacementStep[j], '-------->', f'Dincr = ', Dincr)
        if unconvergeSteps > maxUnconvergedSteps:
            break
        ops.integrator("DisplacementControl", IDctrlNode, 1, Dincr)
        ok = ops.analyze(1)
        D0 = D1  # move to next step
        if ok < 0:
            unconvergeSteps = unconvergeSteps + 1

        finishedSteps = j+1
        disp = ops.nodeDisp(IDctrlNode, 1)
        baseShear = -ops.getLoadFactor(2)
        dataPush[j + 1, 0] = disp
        dataPush[j + 1, 1] = baseShear

    toc = time.time()
    if printProgression:
        print('CYCLIC ANALYSIS DONE IN {:1.2f} seconds'.format(toc - tic))

    if plotPushOverResults:
        plt.rcParams.update({'font.size': 14})
        plt.rcParams["font.family"] = "Times New Roman"

        # Load the data from the files
        force_data = np.loadtxt('RunTimeNodalResults/Cyclic_Horizontal_Reaction.out')
        displacement_data = np.loadtxt('RunTimeNodalResults/Cyclic_Horizontal_Displacement.out')
        # Plot Force vs. Displacement
        plt.figure(figsize=(7, 6), dpi=100)
        # plt.plot(displacement_data, -force_data, color='blue', linewidth=1.2, label='Numerical test')
        plt.plot([row[0] for row in dataPush[:finishedSteps]], [-row[1] for row in dataPush[:finishedSteps]], color="red", linestyle="-", linewidth=1.2, label='Experimental test')
        plt.axhline(0, color='black', linewidth=0.4)
        plt.axvline(0, color='black', linewidth=0.4)
        plt.grid(linestyle='dotted')
        font_settings = {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14}
        plt.xlabel('Displacement (mm)', fontdict=font_settings)
        plt.ylabel('Base Shear (kN)', fontdict=font_settings)
        plt.yticks(fontname='Cambria', fontsize=14)
        plt.xticks(fontname='Cambria', fontsize=14)
        plt.title(f'Specimen', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
        plt.tight_layout()
        plt.legend()

        # if plotValidation:
        #     # Read test output data to plot
        #     Test = np.loadtxt("RunTimeNodalResults/experimental_data.txt", delimiter="\t", unpack="False")
        #     plt.plot(Test[0, :], Test[1, :], color="black", linewidth=0.8, linestyle="--", label='Experimental Data')
        #     plt.xlim(-1, 25)
        #     plt.xticks(np.linspace(-20, 20, 11, endpoint=True))

        # plt.savefig(f'CyclicValidation.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.show()

    # return [dataPush[:, 0], -dataPush[:, 1]]
    return [dataPush[0:finishedSteps, 0], -dataPush[0:finishedSteps, 1]]

