import openseespy.opensees as ops
import numpy as np
import time
import math
import matplotlib.pyplot as plt

# ------------------------------
# Start of model generation
# -----------------------------
mm = 1.0  # 1 millimeter
N = 1.0  # 1 Newton
sec = 1.0  # 1 second
m = 1000.0 * mm  # 1 meter is 1000 millimeters
cm = 10.0 * mm  # 1 centimeter is 10 millimeters
kN = 1000.0 * N  # 1 kilo-Newton is 1000 Newtons
m2 = m * m  # Square meter
cm2 = cm * cm  # Square centimeter
mm2 = mm * mm  # Square millimeter
MPa = N / mm2  # MegaPascal (Pressure)
kPa = 0.001 * MPa  # KiloPascal (Pressure)
GPa = 1000 * MPa  # GigaPascal (Pressure)


def plotting(x_data, y_data, x_label, y_label, title, save_fig=True, plotValidation=True):
    plt.rcParams.update({'font.size': 14, "font.family": ["Cambria", "Times New Roman"]})

    # Plot Force vs. Displacement
    plt.figure(figsize=(7, 6), dpi=100)
    # Read test output data to plot
    if plotValidation:
        Test = np.loadtxt(f"CyclicValidation/{title}.txt", delimiter="\t", unpack="False")
        plt.plot(Test[0, :], Test[1, :], color="black", linewidth=1.0, linestyle="--", label='Experimental Data')

    plt.plot(x_data, y_data, color='blue', linewidth=1.2, label='Numerical Analysis')
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted')
    font_settings = {'fontname': 'Cambria', 'size': 14}
    plt.xlabel(x_label, fontdict=font_settings)
    plt.ylabel(y_label, fontdict=font_settings)
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.title(f"Specimen : {title}", fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.tight_layout()
    plt.legend()

    if save_fig:
        plt.savefig('CyclicValidation/' + title + '.svg', format='svg', dpi=300, bbox_inches='tight')

    plt.show()

def generate_cyclic_load(duration=6.0, sampling_rate=50, max_displacement=75):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    displacement_slope = (max_displacement / 2) / (duration / 2)
    cyclic_load = (displacement_slope * t) * np.sin(2 * np.pi * t)

    return cyclic_load

def RebarArea(RebarDiametermm):
    a = math.pi * (RebarDiametermm / 2) ** 2  # compute area
    return a

def Thomsen_and_Wallace_RW2():
    # https://sci-hub.st/10.1061/(asce)0733-9445(2004)130:4(618)
    global name
    name = 'Thomsen_and_Wallace_RW2'
    # Wall Geometry ------------------------------------------------------------------
    tw = 102 * mm  # Wall thickness
    hw = 3.81 * m  # Wall height
    lw = 1.22 * m  # Wall length
    lbe = 190 * mm  # Boundary element length
    lweb = lw - (2 * lbe)

    # Material proprieties -----------------------------------------------------------
    fc = 41.75 * MPa  # Concrete peak compressive stress (+Tension, -Compression)
    fyb = 434 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fyw = 448 * MPa  # Steel tension yield strength (+Tension, -Compression)

    # ---- Steel in Y direction (BE + Web) -------------------------------------------
    YbeNum = 8  # BE long reinforcement diameter (mm)
    YbeDiam = 9.53  # BE long reinforcement diameter (mm)
    YwebNum = 8  # Web long reinforcement diameter (mm)
    YwebDiam = 6.35  # Web long reinforcement diameter (mm)
    rouYb = (RebarArea(YbeDiam) * YbeNum) / (lbe * tw)  # Y boundary        0.003
    rouYw = (RebarArea(YwebDiam) * YwebNum) / (lweb * tw)  # Y web          0.0293

    # ---- Steel in X direction (BE + Web) -------------------------------------------
    XbeNum = 35 * 4  # BE long reinforcement diameter (mm)
    XbeDiam = 6.35  # BE long reinforcement diameter (mm)
    XwebNum = 20 * 2  # Web long reinforcement diameter (mm)
    XwebDiam = 6.35  # Web long reinforcement diameter (mm)
    rouXb = (RebarArea(XbeDiam) * XbeNum) / (hw * tw)  # Y boundary        0.01
    rouXw = (RebarArea(XwebDiam) * XwebNum) / (hw * tw)  # Y web          0.0033

    loadcoef = 0.0835

    DisplacementStep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41, -42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41, -42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41, -42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -56, -57, -58, -59, -60, -61, -62, -63, -64, -65, -66, -67, -68, -69, -70, -69, -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41, -42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -56, -57, -58, -59, -60, -61, -62, -63, -64, -65, -66, -67, -68, -69, -70, -69, -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41, -42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -56, -57, -58, -59, -60, -61, -62, -63, -64, -65, -66, -67, -68, -69, -70, -71, -72, -73, -74, -75, -76, -77, -78, -79, -80, -81, -82, -83, -84, -85, -84, -83, -82, -81, -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69, -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41, -42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -56, -57, -58, -59, -60, -61, -62, -63, -64, -65, -66, -67, -68, -69, -70, -71, -72, -73, -74, -75, -76, -77, -78, -79, -80, -81, -82, -83, -84, -85, -84, -83, -82, -81, -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69, -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]  # Displacement steps For Thomsen_and_Wallace_RW2

    # DisplacementStep = generate_cyclic_load(duration=10, sampling_rate=50, max_displacement=89)

    return tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadcoef, DisplacementStep


ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)

# ---------------- load geometry and material ------------
validation_model = Thomsen_and_Wallace_RW2()
tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadcoef, DisplacementStep = map(np.float64, validation_model)

# --------------- Discretization of the model ------------
eleH = 16
eleL = 8
wall_thickness = tw  # Wall thickness
wall_height = hw  # Wall height
wall_length = lw  # Wall width
length_be = lbe  # Length of the Boundary Element
length_web = lweb = lw - (2 * lbe)  # Length of the Web
m = eleH
n = eleL
eleBE = 2
eleWeb = eleL - eleBE
elelweb = lweb / eleWeb

# --------------- create nodes ---------------
# Loop through the list of node values
for i in range(1, eleH + 2):
    ops.node(i, 0, (i - 1) * (hw / eleH))

ops.fix(1, 1, 1, 1)  # Fixed condition at node 1

ControlNode = eleH + 1  # Control Node (TopNode)
ControlNodeDof = 1  # Control DOF 1 = X-direction

# ---------------------------------------------------------------------------------------
# Define Steel uni-axial materials
# ---------------------------------------------------------------------------------------
sYb = 1
sYw = 2
sX = 3

# STEEL Y BE (boundary element)
fyYbp = fyb  # fy - tension
fyYbn = fyb  # fy - compression
bybp = 0.0185  # strain hardening - tension
bybn = 0.02  # strain hardening - compression

# STEEL Y WEB
fyYwp = fyw  # fy - tension
fyYwn = fyw  # fy - compression
bywp = 0.025  # strain hardening - tension
bywn = 0.02  # strain hardening - compression

# STEEL X
fyXp = fyw  # fy - tension
fyXn = fyw  # fy - compression
bXp = 0.025  # strain hardening - tension
bXn = 0.02  # strain hardening - compression

# STEEL misc
Es = 200 * GPa  # Young's modulus
R0 = 20.0  # initial value of curvature parameter
Bs = 0.01  # strain-hardening ratio
cR1 = 0.925  # control the transition from elastic to plastic branches
cR2 = 0.015  # control the transition from elastic to plastic branches

# SteelMPF model
ops.uniaxialMaterial('SteelMPF', sYb, fyYbp, fyYbn, Es, bybp, bybn, R0, cR1, cR2)  # Steel Y boundary
ops.uniaxialMaterial('SteelMPF', sYw, fyYwp, fyYwn, Es, bywp, bywn, R0, cR1, cR2)  # Steel Y web
ops.uniaxialMaterial('SteelMPF', sX, fyXp, fyYwn, Es, bXp, bXn, R0, cR1, cR2)  # Steel X
print('--------------------------------------------------------------------------------------------------')
print('SteelMPF', sYb, fyYbp, fyYbn, Es, bybp, bybn, R0, cR1, cR2)  # Steel Y boundary
print('SteelMPF', sYw, fyYwp, fyYwn, Es, bywp, bywn, R0, cR1, cR2)  # Steel Y web
print('SteelMPF', sX, fyYwp, fyYwn, Es, bXp, bXn, R0, cR1, cR2)   # Steel X

# ------------------------------CONCRETE ----------------
concWeb = 4
concBE = 5

# confined
# ----- unconfined concrete for WEB
fc0 = abs(fc)  # Initial concrete strength
Ec0 = 8200.0 * (fc0 ** 0.375)  # Initial elastic modulus
fcU = -fc0  # Unconfined concrete strength
ecU = -(fc0 ** 0.25) / 1152.7  # Unconfined concrete strain
# ecU = -(fc0 ** 0.25) / 28  # Unconfined concrete strain
EcU = Ec0  # Unconfined elastic modulus
ftU = 0.5 * (fc0 ** 0.5)  # Unconfined tensile strength
etU = 2.0 * ftU / EcU  # Unconfined tensile strain
xpU = 2.0
xnU = 2.3
rU = -1.9 + (fc0 / 5.2)

# ----- confined concrete for BE
fl1 = -1.56 * MPa  # Lower limit of confined concrete strength
fl2 = -1.85 * MPa  # Upper limit of confined concrete strength
lam = fl1 / fl2
x = (fl1 + fl2) / (2.0 * fcU)
A = 6.8886 - (0.6069 + 17.275 * lam) * math.exp(-4.989 * lam)
B = (4.5 / (5 / A * (0.9849 - 0.6306 * math.exp(-3.8939 * lam)) - 0.1)) - 5.0
k1 = A * (0.1 + 0.9 / (1 + B * x))
# Check the strength of transverse reinforcement and set k2 accordingly
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

ru = 7.0  # shape parameter - compression
xcrnu = 1.035  # cracking strain - compression
rc = 7.3049  # shape parameter - compression
xcrnc = 1.0125  # cracking strain - compression
et = 0.0002463  # strain at peak tensile stress (0.00008)
rt = 10.2  # shape parameter - tension
xcrp = 100000  # cracking strain - tension

ops.uniaxialMaterial('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, et, rt, xcrp, '-GapClose', 1)  # Web (unconfined concrete)
ops.uniaxialMaterial('ConcreteCM', concBE, fcC, ecC, EcC, rc, xcrnc, ftC, et, rt, xcrp, '-GapClose', 1)  # BE (confined concrete)
print('--------------------------------------------------------------------------------------------------')
print('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, et, rt, xcrp, '-GapClose', 1)  # Web (unconfined concrete)
print('ConcreteCM', concBE, fcC, ecC, EcC, rc, xcrnc, ftC, et, rt, xcrp, '-GapClose', 1)  # BE (confined concrete)
print('--------------------------------------------------------------------------------------------------')

# ----------------------------Shear spring for MVLEM-------------------------------
Ac = lw * tw  # Concrete Wall Area
Gc = EcU / (2 * (1 + 0.2))  # Shear Modulus G = E / 2 * (1 + v)
Kshear = Ac * Gc * (5 / 6)  # Shear stiffness k * A * G ---> k=5/6

ops.uniaxialMaterial('Elastic', 6, Kshear)  # Shear Model for Section Aggregator

Aload = 0.85 * abs(fc) * tw * lw * loadcoef  # Aload = 0.07 * abs(fc) * tw * lw * loadCoeff
print('Axial load (kN) = ', Aload / 1000)

# ---- Steel in Y direction (BE + Web) -------------------------------------------
rouYb = 0.0294884
rouYw = 0.0029699
# ---- Steel in X direction (BE + Web) -------------------------------------------
rouXb = 0.0105450
rouXw = 0.0033169

# ----------------------------FSAM Material -------------------------------
matWeb = 7
matBE = 8
# FSAM model
ops.nDMaterial('FSAM', matWeb, 0.0, sX, sYw, concWeb, rouXw, rouYw, 0.2, 0.012)  # Web (unconfined concrete)
ops.nDMaterial('FSAM', matBE, 0.0, sX, sYb, concBE, rouXb, rouYb, 0.2, 0.012)  # Boundary (confined concrete)
print('--------------------------------------------------------------------------------------------------')
print('FSAM', matWeb, 0.0, sX, sYw, concWeb, rouXw, rouYw, 0.2, 0.012)  # Web (unconfined concrete)
print('FSAM', matBE, 0.0, sX, sYb, concBE, rouXb, rouYb, 0.2, 0.012)  # Boundary (confined concrete)
print('--------------------------------------------------------------------------------------------------')

# ------------------------------------------------------------------------
#  Define 'MVLEM' or  'SFI-MVLEM' elements
# ------------------------------------------------------------------------
# Set 'MVLEM' parameters thick, width, rho, matConcrete, matSteel
MVLEM_thick = [tw] * n
MVLEM_width = [lbe if i in (0, n - 1) else elelweb for i in range(n)]
MVLEM_rho = [rouYb if i in (0, n - 1) else rouYw for i in range(n)]
MVLEM_matConcrete = [concBE if i in (0, n - 1) else concWeb for i in range(n)]
MVLEM_matSteel = [sYb if i in (0, n - 1) else sYw for i in range(n)]

MVLEM_mat = [matBE if i in (0, n - 1) else matWeb for i in range(n)]

for i in range(eleH):
    # ------------------ MVLEM ----------------------------------------------
    ops.element('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', 6)
    print('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', 6)

    # ---------------- SFI_MVLEM -------------------------------------------
   # ops.element('SFI_MVLEM', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)
   # print('SFI_MVLEM', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)

# ------------------------------
# Start of analysis generation
# ------------------------------
# -------------GRAVITY-----------------
steps = 10
print("RUNNING GRAVITY ANALYSIS")
ops.timeSeries('Linear', 1, '-factor', 1.0)  # create TimeSeries for gravity analysis
ops.pattern('Plain', 1, 1)
ops.load(ControlNode, *[0.0, -Aload, 0.0])  # apply vertical load
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.test('NormDispIncr', 1.0e-5, 100, 0)
ops.algorithm('Newton')
ops.integrator('LoadControl', 1 / steps)
ops.analysis('Static')
ops.analyze(steps)
ops.loadConst('-time', 0.0)  # hold gravity constant and restart time for further analysis
print("GRAVITY ANALYSIS DONE!")

# -------------CYCLIC-----------------
print("RUNNING CYCLIC ANALYSIS")
ops.timeSeries('Linear', 2, '-factor', 1.0)
ops.pattern('Plain', 2, 2)
ops.load(ControlNode, *[1.0, 0.0, 0.0])  # apply vertical load
ops.constraints('Penalty', 1e20, 1e20)  # Transformation 'Penalty', 1e20, 1e20
ops.numberer('RCM')
ops.system("BandGen")
ops.test('NormDispIncr', 1e-6, 1000, 0)
ops.algorithm('Newton')

# Define analysis parameters
maxUnconvergedSteps = 10
unconvergeSteps = 0
Nsteps = len(DisplacementStep)
finishedSteps = 0
dispData = np.zeros(Nsteps + 1)
baseShearData = np.zeros(Nsteps + 1)

test = {1: 'NormDispIncr', 2: 'RelativeEnergyIncr', 4: 'RelativeNormUnbalance', 5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
algorithm = {1: 'KrylovNewton', 2: 'SecantNewton', 4: 'RaphsonNewton', 5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

# Perform cyclic analysis
D0 = 0.0
for j in range(Nsteps):
    D1 = DisplacementStep[j]
    Dincr = D1 - D0
    print(f'Step {j} -------->', f'Dincr = ', Dincr)
    # if unconvergeSteps > maxUnconvergedSteps:
    #     break
    ops.integrator("DisplacementControl", ControlNode, 1, Dincr)
    ops.analysis('Static')
    ok = ops.analyze(1)
    # if ok != 0:
    #     # If not converged, reduce the increment
    #     unconvergeSteps += 1
    #     ts = 20
    #     smallDincr = Dincr / ts  # Try 10x smaller increments
#
    #     for k in range(1, ts):
    #         print(f'Small Step {k} -------->', f'smallDincr = ', smallDincr)
    #         ops.integrator("DisplacementControl", ControlNode, 1, smallDincr)
    #         ops.analysis('Static')
    #         ok = ops.analyze(1)
    if ok != 0:
        for i in test:
            for k in algorithm:

                if ok != 0:
                    if k < 4:
                        ops.algorithm(algorithm[k], '-initial')

                    else:
                        ops.algorithm(algorithm[k])

                    ops.test(test[i], 1e-8, 1000)
                    ok = ops.analyze(1)
                    print(test[i], algorithm[k], ok)
                    if ok == 0:
                        break
                else:
                    continue
    D0 = D1  # move to next step

    finishedSteps = j + 1
    disp = ops.nodeDisp(ControlNode, 1)
    baseShear = -ops.getLoadFactor(2) / 1000  # Convert to from N to kN
    dispData[j + 1] = disp
    baseShearData[j + 1] = baseShear

    print(f'\033[92m InputDisplacement {j} = {DisplacementStep[j]}\033[0m')
    print(f'\033[91mOutputDisplacement {j} = {dispData[j + 1]}\033[0m')

print('CYCLIC ANALYSIS DONE')

plotting(dispData[0:finishedSteps], -baseShearData[0:finishedSteps], 'Displacement (mm)', 'Base Shear (kN)', f'{name}', save_fig=True, plotValidation=True)