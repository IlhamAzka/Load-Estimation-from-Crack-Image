import numpy as np

from scipy.optimize import curve_fit

def fractal_dimension(image:np.ndarray, start, end, shape, iterations=20):
    pixels = find_pixel(image) 
    Ns, scales = box_count(pixels, shape, start=start, end=end, iterations=iterations)

    coeff = np.polyfit(np.log(1 / scales), np.log(Ns), 1)
    return coeff, scales, Ns

# TO CALCULATE:
# - singularity exponent
# - corresponding fractal dimension

# Return crack density of each box on all box scale size
def crack_density_eachbox(image:np.ndarray, start, end, shape, iterations=20):
    X, Y = shape
    pixels = find_pixel(image)
    Ns, scales = box_count(pixels, shape, start=start, end=end, iterations=iterations)

    # range value of q [-5, +5]
    # distortion_param = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    crack_density_all = []

    for scale in scales:
        H, _ = np.histogramdd(pixels, bins=(np.arange(0, X, scale), np.arange(0, Y, scale)))
        H_len = len(H)

        # Crack density
        total_crack_pixel = np.sum(H)
        crack_density = H / total_crack_pixel        
        crack_density_all.append(crack_density)

    return crack_density_all

def crack_density_norm(crack_density_distort):
    cdn_total = []

    for cdd in crack_density_distort:
        total_crack_density_distort = np.sum(cdd)
        # print(f"cdd element: {cdd}")
        # print(f"tcdd: {total_crack_density_distort}")
        cdn = cdd / total_crack_density_distort
        cdn_total.append(cdn)

    # print(f"cdn_total: {len(cdn_total)}")

    return cdn_total

def singularity_exponent(crack_density, scales):
    distortion_param = np.linspace(start=-5, stop=5, num=50)
    singularity_exponent_all = []
    for q in distortion_param:
        cdd = []
        for cd in crack_density:
            temp = np.power(cd[cd > 0], q)
            temp = np.nan_to_num(temp, nan=0.0)
            cdd.append(temp)

        cdn = crack_density_norm(cdd)
        upper_part = []
        for i in range(len(cdn)):
            up = np.sum(cdn[i] * np.log(cdd[i]))
            upper_part.append(up)

        # print(f"scales: {scales}")
        se = np.polyfit(np.log(scales), upper_part, 1)
        singularity_exponent_all.append(se)

    return singularity_exponent_all

def general_fractal_dimension(crack_density, scales):
    distortion_param = np.linspace(start=-5, stop=5, num=50)
    fd_all = []
    for q in distortion_param:
        cdd = []
        for cd in crack_density:
            temp = np.power(cd[cd > 0], q)
            temp = np.nan_to_num(temp, nan=0.0)
            cdd.append(temp)

        cdn = crack_density_norm(cdd)
        upper_part = []
        for e in cdn:
            up = np.sum(e * np.log(e))
            upper_part.append(up)

        fd = np.polyfit(np.log(scales), upper_part, 1)
        fd_all.append(fd)

    return fd_all

def _power(x, a, b):
    return b * x**a

def find_pixel(image:np.ndarray):
    # find black pixel position from all elements 
    # that laid as 2d matrix in original matrix  
    return np.argwhere(image == 1)

def box_count(pixels, shape, start=1, end=8, iterations=20):
    X, Y = shape

    end = min(end, np.log2(min(X, Y)))

    # scales determine the box size / grid size 
    # start
    scales = np.logspace(start, end, num=iterations, endpoint=True, base=2)

    Ns = []
    for scale in scales:
        # With histogramdd, we are counting the black pixel
        # that present in a grid with sizes bins
        H, _ = np.histogramdd(pixels, bins=(np.arange(0, X, scale), np.arange(0, Y, scale)))
        # print(H.shape)
        # print(H)

        # We check if black pixel present in a grid
        # and then count the grid that satisfy the condition
        Ns.append(np.sum(H > 0))

    return Ns, scales