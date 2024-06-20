import openseespy.opensees as ops
import numpy as np
import time
import math
import matplotlib.pyplot as plt


def generate_cyclic_load():
    t = np.linspace(0, 6, int(100 * 6), endpoint=False)
    displacement_slope = (85 / 2) / (6 / 2)
    cyclic_load = (displacement_slope * t) * np.sin(2 * np.pi * t)
    return cyclic_load


DisplacementStep = generate_cyclic_load()

ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)

ops.node(1, 0, 0.0)
ops.node(2, 0, 732.0)
ops.node(3, 0, 1464.0)
ops.node(4, 0, 2196.0)
ops.node(5, 0, 2928.0)
ops.node(6, 0, 3660.0)

ops.fix(1, 1, 1, 1)

IDctrlNode = 6
IDctrlDOF = 1

# STEEL Reinforcing steel
ops.uniaxialMaterial('SteelMPF', 1, 434.0, 434.0, 200000.0, 0.01, 0.01, 20.0, 0.925, 0.15)  # steel Y web
ops.uniaxialMaterial('SteelMPF', 2, 448.0, 448.0, 200000.0, 0.02, 0.02, 20.0, 0.925, 0.15)  # steel Y boundary
ops.uniaxialMaterial('SteelMPF', 3, 448.0, 448.0, 200000.0, 0.02, 0.02, 20.0, 0.925, 0.15)  # steel X

# CONCRETE	-------------------------------------------------------------------------
ops.uniaxialMaterial('ConcreteCM', 4, -50.04, -0.0028, 35000.0, 7.3049, 1.0125, 2.05, 8e-05, 1.2, 10000, '-GapClose', 0)
ops.uniaxialMaterial('ConcreteCM', 5, -41.7, -0.0023, 35000.0, 7, 1.039, 2.05, 8e-05, 1.2, 10000, '-GapClose', 0)

# Shear spring -----------------------------------------------------------
ops.uniaxialMaterial('Elastic', 6, 1512291666)

# ---- Steel in Y and X direction  -------------------------------------------
rouYb = 0.02948
rouYw = 0.00376
rouXb = 0.00282
rouXw = 0.00282
ops.nDMaterial('FSAM', 7, 0.0, 3, 1, 4, rouXb, rouYb, 0.2, 0.012)
ops.nDMaterial('FSAM', 8, 0.0, 3, 2, 5, rouXw, rouYw, 0.2, 0.012)

#  Define 'SFI_MVLEM' elements
ops.element('SFI_MVLEM', 1, 1, 2, 5, 0.4, '-thick', 102.0, 102.0, 102.0, 102.0, 102.0, '-width', 190.0, 280.0, 280.0, 280.0, 190.0, '-mat', 7, 8, 8, 8, 7)
ops.element('SFI_MVLEM', 2, 2, 3, 5, 0.4, '-thick', 102.0, 102.0, 102.0, 102.0, 102.0, '-width', 190.0, 280.0, 280.0, 280.0, 190.0, '-mat', 7, 8, 8, 8, 7)
ops.element('SFI_MVLEM', 3, 3, 4, 5, 0.4, '-thick', 102.0, 102.0, 102.0, 102.0, 102.0, '-width', 190.0, 280.0, 280.0, 280.0, 190.0, '-mat', 7, 8, 8, 8, 7)
ops.element('SFI_MVLEM', 4, 4, 5, 5, 0.4, '-thick', 102.0, 102.0, 102.0, 102.0, 102.0, '-width', 190.0, 280.0, 280.0, 280.0, 190.0, '-mat', 7, 8, 8, 8, 7)
ops.element('SFI_MVLEM', 5, 5, 6, 5, 0.4, '-thick', 102.0, 102.0, 102.0, 102.0, 102.0, '-width', 190.0, 280.0, 280.0, 280.0, 190.0, '-mat', 7, 8, 8, 8, 7)

# -------------GRAVITY-----------------
steps = 10
ops.timeSeries('Linear', 1, '-factor', 1.0)  # create TimeSeries for gravity analysis
ops.pattern('Plain', 1, 1)
ops.load(IDctrlNode, *[0.0, -441077, 0.0])  # apply vertical load
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.test('NormDispIncr', 1.0e-6, 100, 0)
ops.algorithm('Newton')
ops.integrator('LoadControl', 1 / steps)
ops.analysis('Static')
ops.analyze(steps)
ops.loadConst('-time', 0.0)

# define parameters for adaptive time-step
max_factor = 1.0  # 1.0 -> don't make it larger than initial time step
min_factor = 1e-06  # at most initial/1e6
max_factor_increment = 1.5  # define how fast the factor can increase
min_factor_increment = 1e-06  # define how fast the factor can decrease
max_iter = 5000
desired_iter = int(max_iter / 2)  # should be higher then the desired number of iterations

# -------------CYCLIC-----------------
ops.timeSeries('Linear', 2, '-factor', 1.0)
ops.pattern('Plain', 2, 2)
RefLoad = 1000e3
ops.load(IDctrlNode, *[RefLoad, 0.0, 0.0])
ops.constraints('Transformation')  # Transformation 'Penalty', 1e20, 1e20
ops.numberer('RCM')
ops.system("BandGen")
ops.test('NormDispIncr', 1e-8, desired_iter, 0)
ops.algorithm('KrylovNewton')

Nsteps = len(DisplacementStep)
finishedSteps = 0
dispData = np.zeros(Nsteps + 1)
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
            print("Target displacement has been reached. Current Dincr = {:.3g}".format(dU_cumulative))
            increment_done = True
            break
        # adapt the current displacement increment
        dU_adapt = dU * factor
        if abs(dU_cumulative + dU_adapt) > (abs(Dincr) - dU_tolerance):
            dU_adapt = Dincr - dU_cumulative
        # update integrator
        ops.integrator("DisplacementControl", IDctrlNode, 1, dU_adapt)
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
                print("Increasing increment factor due to faster convergence. Factor = {:.3g}".format(factor))
            old_factor = factor
            dU_cumulative += dU_adapt
        else:
            num_iter = max_iter
            factor_increment = max(min_factor_increment, desired_iter / num_iter)
            factor *= factor_increment
            print("Reducing increment factor due to non convergence. Factor = {:.3g}".format(factor))
            if factor < min_factor:
                print("ERROR: current factor is less then the minimum allowed ({:.3g} < {:.3g})".format(factor, min_factor))
                print("ERROR: the analysis did not converge")
                break
    if not increment_done:
        break
    else:
        D0 = D1  # move to next step

    finishedSteps = j + 1
    disp = ops.nodeDisp(IDctrlNode, 1)
    baseShear = -ops.getLoadFactor(2) / 1000 * RefLoad  # Convert to from N to kN
    dispData[j + 1] = disp
    ShearData[j + 1] = baseShear

plt.rcParams.update({'font.size': 10, "font.family": ["Times New Roman", "Cambria"]})
plt.figure(figsize=(4 * 1.1, 3 * 1.25))
plt.plot(dispData, -ShearData, color='red', linewidth=1.2)
plt.axhline(0, color='black', linewidth=0.4)
plt.axvline(0, color='black', linewidth=0.4)
plt.grid(linestyle='dotted')
font_settings = {'fontname': 'Times New Roman', 'size': 10}
plt.xlabel('Displacement', fontdict=font_settings)
plt.ylabel('Shear', fontdict=font_settings)
plt.legend(fontsize='small')
plt.show()