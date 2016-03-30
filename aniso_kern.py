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
    b = w+w+1
    m = np.zeros((b, b), dtype=np.float)
    d = (w-1) / 2
    m[w, w] = 1.0
    #FIXME: document why this is correct
    rotated = scipy.ndimage.rotate(scipy.ndimage.gaussian_filter(m, sigmas), angle)
    #print "rotated size is", rotated.shape
    #print rotated
    row, col = divmod(rotated.argmax(), rotated.shape[0])
    #print row, col
    return rotated[row-d:row+d+1,col-d:col+d+1]

if __name__ == '__main__':
    import sys

    np.set_printoptions(linewidth=1000, precision=3)

    try:
        w, sigma1, sigma2, angle = [float(s) for s in sys.argv[1:]]
    except ValueError:
        print "Use: %s width sigma1 sigma2 angle" % sys.argv[0]
        sys.exit(1)

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
