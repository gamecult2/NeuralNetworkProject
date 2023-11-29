import math as math


def hertz_impact_material(e=0.7, r1=0.064, fmax=250000, v0 = 4.8, t = 0.005):
    miu1 = 0.3  # Impactor possion ratio
    miu2 = 0.2  # Beam possion ratio
    e1 = 1.5 * 10 ** 5 * 10 ** 6  # Impactor modulus (in Pa)
    e2 = 3.25 * 10 ** 4 * 10 ** 6  # Beam modulus (in Pa)
    # print(e2)
    lamda1 = (1 - miu1 ** 2) / e1
    lamda2 = (1 - miu2 ** 2) / e2
    kh = 4 / 3 * math.sqrt(r1) * (1 / (lamda1 + lamda2))  # Hertz contact stiffness
    # Calculate deltam (maximum penetration)
    deltam = (fmax / kh) ** (2 / 3)
    deltae = (kh * deltam ** 2.5 * (1 - e ** 2)) / 2.5
    keff = kh * math.sqrt(deltam)
    kt1 = keff + (deltae / (0.1 * deltam ** 2))
    kt2 = keff - (deltae / (0.9 * deltam ** 2))
    kt1 = kt1 / 1000
    kt2 = kt2 / 1000
    deltam = deltam * 1000
    deltay = - 0.1 * deltam
    gap = - 1000 * v0 * t / 2
    return kt1, kt2, deltay, gap
