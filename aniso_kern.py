#!/usr/bin/python

import numpy as np
import scipy.ndimage

def generate_aniso_gauss_kernel(w, sigmas, angle):
    """Compute anisotropic Gaussian filter.

    w   is width of array (odd positive integer)

    sigmas  - tuple

    angle   - in degrees

    Returns
    -------

    kernel

    """
    m = np.zeros((w, w), dtype=np.float)
    d = (w-1) / 2
    m[d, d] = 1.0
    return scipy.ndimage.rotate(scipy.ndimage.gaussian_filter(m, sigmas), angle)

if __name__ == '__main__':
    import sys

    w, sigma1, sigma2, angle = [float(s) for s in sys.argv[1:]]

    w = int(w)

    m = generate_aniso_gauss_kernel(w, (sigma1, sigma2), angle)
    #print m
    
    with open("aniso-%dx%d_%g_%g_%g.kern" % (w, w, sigma1, sigma2, angle), "w") as fd:
        fd.write("np.array([")
        for i in range(w):
            fd.write('[')
            for j in range(w):
                fd.write("%g, " % m[i, j])
            fd.write('],\n')
        fd.write('])')
