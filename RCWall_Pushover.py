import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import openseespy.postprocessing.ops_vis as opsv
import openseespy.postprocessing.Get_Rendering as opsplt
import time
import math


def RebarArea(RebarDiametermm):
    r = (RebarDiametermm / 2)
    a = 3.1416 * r ** 2  # compute area
    return a

# ------------------------------------------------------------------------
# Define units - All results will be in { mm, N, MPa and Sec }
# ------------------------------------------------------------------------
mm = 1.0            # 1 millimeter
N = 1.0             # 1 Newton
sec = 1.0           # 1 second

m = 1000.0 * mm     # 1 meter is 1000 millimeters
cm = 10.0 * mm      # 1 centimeter is 10 millimeters
kN = 1000.0 * N     # 1 kilo-Newton is 1000 Newtons
m2 = m * m          # Square meter
cm2 = cm * cm       # Square centimeter
mm2 = mm * mm       # Square millimeter
MPa = N / mm2       # MegaPascal (Pressure)
kPa = 0.001 * MPa   # KiloPascal (Pressure)
GPa = 1000 * MPa    # GigaPascal (Pressure)

# ------------------------------------------------------------------------
# Define Validation Example as Parameter Values
# ------------------------------------------------------------------------
# Wall Geometry
hw = 4.56 * m          # Wall height
lw = 2.00 * m          # Wall length
tw = 150 * mm           # Wall thickness
lbe = 150 * mm         # Boundary element length

# Material proprieties
fc = -50.6 * MPa       # Concrete peak compressive stress (+Tension, -Compression)
fy = 470 * MPa         # Steel tension yield strength (+Tension, -Compression)
bereinfratio = 2.5     # BE long reinforcement ratio (percentage)
webreinfratio = 1.375  # Web long reinforcement ratio (percentage)

bereinfDiam = 10        # BE long reinforcement diameter (mm)
bereinfNum = 6         # BE long reinforcement diameter (mm)

webreinfDiam = 6       # Web long reinforcement diameter (mm)
webreinfNum = 24       # Web long reinforcement diameter (mm)

loadcoef = 0.05
rouYb = 0.033
rouYw = 0.025

def build_model(hw, lw, tw, lbe, fc, fy, rouYb, rouYw, loadcoef,
                eleH=16,
                eleL=8,
                performPushOver=True):
    ops.wipe()  # Clear opensees model
    ops.model('basic', '-ndm', 2, '-ndf', 3)  # Model of 2 dimensions, 3 dof per node

    # ------------------------------------------------------------------------
    # Set geometry, ops.nodes, boundary conditions
    # ------------------------------------------------------------------------
    # Wall Geometry
    wall_height = hw     # Wall height
    wall_length = lw     # Wall width
    wall_thickness = tw  # Wall thickness
    length_be = lbe
    length_web = lweb = lw - (2 * lbe)

    # ------------------------------------------------------------------------
    # discretization of the wall geometry
    # ------------------------------------------------------------------------
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
    IDctrlDOF = 1          # Control DOF 1 = X-direction

    # ------------------------------------------------------------------------
    # Define uni-axial materials
    # ------------------------------------------------------------------------
    # STEEL Y boundary element
    fyYbp = fy   # fy - tension
    fyYbn = fyYbp       # fy - compression
    bybp = 0.01         # strain hardening - tension
    bybn = 0.02         # strain hardening - compression

    # STEEL Y web
    fyYwp = fy   # fy - tension
    fyYwn = fyYwp       # fy - compression
    bywp = 0.035        # strain hardening - tension
    bywn = 0.02         # strain hardening - compression

    # STEEL misc
    Es = 200.6 * GPa   # Young's modulus
    R0 = 20.0          # initial value of curvature parameter
    a1 = 0.925         # curvature degradation parameter
    a2 = 0.0015        # curvature degradation parameter
    Bs = 0.01          # strain-hardening ratio
    cR1 = 0.925        # control the transition from elastic to plastic branches
    cR2 = 0.15         # control the transition from elastic to plastic branches

    # Build steel materials
    # ops.uniaxialMaterial('SteelMPF', 1, fyYbp, fyYbn, Es, bybp, bybn, R0, a1, a2)  # steel Y boundary
    # ops.uniaxialMaterial('SteelMPF', 2, fyYwp, fyYwn, Es, bywp, bywn, R0, a1, a2)  # steel Y web
    # ops.uniaxialMaterial('MinMax', 6, 1, '-min', -0.06, '-max', 0.06)
    # ops.uniaxialMaterial('MinMax', 7, 2, '-min', -0.06, '-max', 0.06)

    # STEEL Reinforcing steel
    ops.uniaxialMaterial('Steel02', 1, fyYbp, Es, Bs, 20.0, 0.925, 0.15)  # steel Y boundary
    ops.uniaxialMaterial('Steel02', 2, fyYwp, Es, Bs, 20.0, 0.925, 0.15)  # steel Y web
    ops.uniaxialMaterial('MinMax', 6, 1, '-min', -0.06, '-max', 0.06)
    ops.uniaxialMaterial('MinMax', 7, 2, '-min', -0.06, '-max', 0.06)
    '''
    # CONCRETE misc ---------------------------------------------------------------
    # fc = -42.8 * MPa     # Maximum compressive stress
    Ec = 29 * GPa     # Young's modulus

    # unconfined
    fpc = fc                 # peak compressive stress
    ec0 = 2 * fpc / Ec       # strain at maximum compressive stress (-0.0021)
    Ec = 31.03 * GPa         # Young's modulus
    ru = 7                   # shape parameter - compression
    xcrnu = 1.039            # cracking strain - compression

    # confined
    Ecc = Ec                 # Young's modulus
    fpcc = 1.112 * fc * MPa  # peak compressive stress
    ec0c = 2 * fpcc / Ecc    # strain at maximum compressive stress (-0.0033)
    rc = 7.3049              # shape parameter - compression
    xcrnc = 1.0125           # cracking strain - compression
    ft = 2.03 * MPa          # peak tensile stress
    et = 0.00008             # strain at peak tensile stress
    rt = 1.2                 # shape parameter - tension
    xcrp = 10000             # cracking strain - tension

    # Build concrete materials
    ops.uniaxialMaterial('ConcreteCM', 3, fpc, ec0, Ec, ru, xcrnu, ft, et, rt, xcrp)     # unconfined concrete
    ops.uniaxialMaterial('ConcreteCM', 4, fpcc, ec0c, Ecc, rc, xcrnc, ft, et, rt, xcrp)  # confined concrete

    ''' 
    # CONCRETE ---------------------------------------------------------------
    # fc = fc     # Concrete Compressive Strength, MPa   (+Tension, -Compression)
    Ec = 29 * GPa     # Concrete Elastic Modulus
    # unconfined concrete
    fc1U = fc            # unconfined concrete (todeschini parabolic model), maximum stress
    eps1U = -0.0021      # strain at maximum strength of unconfined concrete
    fc2U = 0.2 * fc1U    # ultimate stress
    eps2U = -0.01        # strain at ultimate stress
    lam = 0.1            # ratio between unloading slope at $eps2 and initial slope $Ec
    # confined concrete
    fc1C = fc            # confined concrete (mander model), maximum stress
    eps1C = 2 * fc1C/Ec  # strain at maximum stress (-0.0033)
    fc2C = 0.2 * fc1C    # ultimate stress
    eps2C = 5 * eps1C    # strain at ultimate stress
    # tensile-strength properties
    ftC = 2.03 * MPa     # tensile strength +tension
    ftU = 2.03 * MPa     # tensile strength +tension
    Ets = ftU / 0.002    # tension softening stiffness

    ops.uniaxialMaterial('Concrete02', 3, fc1U, eps1U, fc2U, eps2U, lam, ftU, Ets)  # COVER CONCRETE  (unconfined)
    ops.uniaxialMaterial('Concrete02', 4, fc1C, eps1C, fc2C, eps2C, lam, ftC, Ets)  # CORE CONCRETE  (confined)
    # print('Concrete02', 3, fc1U, eps1U, fc2U, eps2U, lam, ftU, Ets)  # COVER CONCRETE  (unconfined)
    # print('Concrete02', 4, fc1C, eps1C, fc2C, eps2C, lam, ftC, Ets)  # CORE CONCRETE  (confined)
    #'''  # Other Concrete02 Model

    # SHEAR -----------------------------------------------------------------
    # NOTE: large shear stiffness assigned since only flexural response
    Ac = lw * tw                # Concrete Wall Area
    Gc = Ec / (2 * (1 + 0.2))   # Shear Modulus G = E / 2 * (1 + v)
    Kshear = Ac * Gc * (5 / 6)  # Shear stiffness k * A * G ---> k=5/6

    # Build shear material to assign for MVLEM element
    ops.uniaxialMaterial('Elastic', 5, Kshear)  # Shear Model for Section Aggregator CONCRETE C40

    # axial force in N according to ACI318-19 (not considering the reinforced steel at this point for simplicity)
    global Pforce
    # Pforce = 0.07 * fpc * lw * tw
    Pforce = 0.85 * abs(fc) * tw * lw * loadcoef
    print('Pforce = ', Pforce/1000)
    # ------------------------------------------------------------------------
    #  Calculate the parameters for 'MVLEM' elements
    # ------------------------------------------------------------------------
    # Calculate 'MVLEM' elements reinforcement Ratios (RebarArea / ConcreteArea)
    rouYb = (RebarArea(bereinfDiam) * bereinfNum) / (lbe * tw)        # Y boundary
    rouYw = (RebarArea(webreinfDiam) * webreinfNum) / (elelweb * tw)  # Y web
    rouYw = rouYw / eleWeb
    # rouYb = bereinfratio / 100
    # rouYw = webreinfratio / 100
    print('rouYb = ', rouYb)
    print('rouYw = ', rouYw)

    # Set 'MVLEM' parameters thick, width, rho, matConcrete, matSteel
    MVLEM_thick = [tw] * n
    MVLEM_width = [lbe if i in (0, n - 1) else elelweb for i in range(n)]
    MVLEM_rho = [rouYb if i in (0, n - 1) else rouYw for i in range(n)]
    MVLEM_matConcrete = [4 if i in (0, n - 1) else 3 for i in range(n)]
    MVLEM_matSteel = [1 if i in (0, n - 1) else 2 for i in range(n)]

    # ------------------------------------------------------------------------
    #  Define 'MVLEM' elements
    # ------------------------------------------------------------------------
    # Set 'MVLEM' element
    for i in range(eleH):
        ops.element('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', 5)
        # print('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', 5)

    parameters = {
        "wall_height": wall_height,
        "wall_length": wall_length,
        "wall_thickness": wall_thickness,
        "length_be": length_be,
        "length_web": length_web,
        "rouYb": rouYb,
        "rouYw": rouYw,
        "IDctrlNode": IDctrlNode,
        "IDctrlDOF": IDctrlDOF,
        "Pforce": Pforce,
        "eleL": eleL,
        "eleH": eleH,
        "eleBE": eleBE,
        "eleWeb": eleWeb
    }
    return parameters


def run_gravity(steps=10):
    print("RUNNING GRAVITY ANALYSIS")
    ops.recorder('Node', '-file', 'RunTimeNodalResults/Disp.txt', '-closeOnWrite', '-time', '-node', IDctrlNode, '-dof', 2, 'disp')
    ops.recorder('Node', '-file', 'RunTimeNodalResults/Gravity_Reactions.out', '-time', '-node', *[1], '-dof', *[1, 2, 3], 'reaction')

    ops.timeSeries('Linear', 1)  # create TimeSeries for gravity analysis
    ops.pattern('Plain', 1, 1)
    ops.load(IDctrlNode, *[0.0, -Pforce, 0.0])  # apply vertical load

    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-6, 100, 0)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 1 / steps)
    ops.analysis('Static')
    ops.analyze(steps)
    print("GRAVITY ANALYSIS DONE!")

    # Keep the gravity loads for further analysis
    ops.loadConst('-time', 0.0)  # hold gravity constant and restart time

def run_pushover(steps=200, MaxDisp=15, DispIncr=0.1, plotDeformedGravity=False, plotPushOverResults=True, progressBar=None, printProgression=True, recordResults=False):

    print("RUNNING PUSHOVER ANALYSIS")
    tic = time.time()

    ops.recorder('Node', '-file', 'RunTimeNodalResults/Pushover_Horizontal_Reactions.out', '-node', 1, '-dof', 1, 'reaction')
    ops.recorder('Node', '-file', 'RunTimeNodalResults/disp_pushover.out', '-node', IDctrlNode, '-dof', 1, 'disp')

    dataPush = []
    # Apply lateral load based on first mode shape in x direction (EC8-1)
    ops.timeSeries('Linear', 2)  # create TimeSeries for gravity analysis
    ops.pattern('Plain', 2, 2)

    NstepsPush = int(MaxDisp / DispIncr)

    if printProgression:
        print("Starting pushover analysis...")
        print("   total steps: ", NstepsPush)

    ops.load(IDctrlNode, *[1.0, 0.0, 0.0])  # Apply a unit reference load in DOF=1 (nd    FX  FY  MZ)

    ops.constraints('Transformation')
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test('NormDispIncr', 1e-6, 100, 0)
    ops.algorithm('Newton')
    ops.integrator("DisplacementControl", IDctrlNode, 1, DispIncr)  # Target node is IDctrlNode and dof is 1
    ops.analysis("Static")

    maxUnconvergedSteps = 10
    unconvergeSteps = 0
    finishedSteps = 0
    dataPush = np.zeros((NstepsPush + 1, 2))

    # Performs the analysis
    # ops.analyze(NstepsPush)

    # Perform pushover analysis
    for j in range(NstepsPush):
        if unconvergeSteps > maxUnconvergedSteps:
            break

        ok = ops.analyze(1)

        if ok < 0:
            unconvergeSteps = unconvergeSteps + 1

        finishedSteps = j
        disp = ops.nodeDisp(IDctrlNode, 1)
        baseShear = -ops.getLoadFactor(2) * 0.001  # convert to KN
        dataPush[j + 1, 0] = disp
        dataPush[j + 1, 1] = baseShear

        if printProgression:
            print("step", j + 1, "/", NstepsPush, "   ", "disp", "=", str(round(disp, 2)))

    toc = time.time()
    print('PUSHOVER ANALYSIS DONE IN {:1.2f} seconds'.format(toc - tic))

    if plotPushOverResults:
        plt.rcParams.update({'font.size': 14})
        plt.rcParams["font.family"] = "Times New Roman"

        plt.figure(figsize=(6, 4), dpi=100)
        plt.plot(dataPush[0:finishedSteps, 0], -dataPush[0:finishedSteps, 1], color="red", linewidth=1.2, linestyle="-", label='Pushover Analysis')
        plt.axhline(0, color='black', linewidth=0.4)
        plt.axvline(0, color='black', linewidth=0.4)
        plt.grid(linestyle='dotted')
        plt.xlabel('Displacement (mm)')
        plt.ylabel('Base Shear (kN)')

        # if plotValidation:
        #     # Read test output data to plot
        #     Test = np.loadtxt("RunTimeNodalResults/experimental_data.txt", delimiter="\t", unpack="False")
        #     plt.plot(Test[0, :], Test[1, :], color="black", linewidth=0.8, linestyle="--", label='Experimental Data')
        #     plt.xlim(-1, 25)
        #     plt.xticks(np.linspace(-20, 20, 11, endpoint=True))

        plt.tight_layout()
        plt.legend()
        plt.show()

    return [dataPush[0:finishedSteps, 0], -dataPush[0:finishedSteps, 1]], ops


def reset_analysis():
    """
    Resets the analysis by setting time to 0,
    removing the recorders and wiping the analysis.
    """
    ##  Set the time in the Domain to zero
    ops.setTime(0.0)
    # Set the loads constant in the domain
    ops.loadConst()
    # Remove all recorder objects.
    ops.remove('recorders')
    # destroy all components of the Analysis object
    ops.wipeAnalysis()


build_model(hw, lw, tw, lbe, fc, fy, rouYb, rouYw, loadcoef)
run_gravity()
run_pushover()
reset_analysis()
ops.wipe()
