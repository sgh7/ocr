#!/usr/bin/python

import numpy as np
import math
import scipy.ndimage
import matplotlib.pyplot as plt
from StringIO import StringIO

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

# lifted from scipy.ndimage.gaussian_filter1d
def gaussian_filter1d_weights(sigma):
    """One-dimensional 0-order Gaussian filter.

    Parameters
    ----------
    sigma : scalar
        standard deviation for Gaussian kernel
    """
    sd = float(sigma)
    # make the length of the filter equal to 4 times the standard
    # deviations:
    lw = int(4.0 * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum

    return weights

def gaussian_filter1d_weights_offset(sigma, offset=0.0):
    """One-dimensional 0-order Gaussian filter.

    Parameters
    ----------
    sigma : scalar
        standard deviation for Gaussian kernel
    """
    assert abs(offset) < 1.0

    sd = float(sigma)
    # make the length of the filter equal to 4 times the standard
    # deviations:
    lw = int(4.0 * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    sum = 0.0
    sd = sd * sd
    for ii in range(2 * lw + 1):
        arg = ii + offset - lw
        tmp = math.exp(-0.5 * arg*arg / sd)
        weights[ii] = tmp
        sum += tmp

    for ii in range(2 * lw + 1):
        weights[ii] /= sum

    return weights

def gaussian_filter1d_origin(width, sigma, origin=0):
    m = np.zeros((width,), dtype=np.float)
    m[(width-1)/2] = 1.0
    weights = gaussian_filter1d_weights(sigma)
    return scipy.ndimage.correlate1d(m, weights, origin=origin)

def generate_sum_gauss(gp, ortho_sigma, angle, plot_unrot_kernel=False, plot_rot_kernel=False, verbose=False):
    """Compute sum of colinear anisotropic Gaussian filters.

    gp  is an Nx3 array of parameters for the Gaussian functions
        to be summed along the principal axis
    ortho_sigma is the standard deviation for the spread
        along the secondary axis
    angle is the angle in degrees to rotate the result

    Returns
    -------

    kernel

    """

    # 
    wg = (4.0*gp['sigma']+0.5).astype(int)   # widths for each Gaussian out past 4 std deviations
    f_min = (gp['b']-wg).min()
    f_max = (gp['b']+wg).max()
    eval_min = int(f_min)
    eval_max = int(f_max)
    b = 1+eval_max-eval_min
    if b % 2 == 0:
        # force width to an odd integer
        if abs(f_min-eval_min) < abs(f_max-eval_max):
            eval_min -= 1
        else:
            eval_max += 1
        b += 1

    x = np.arange(eval_min, eval_max+1)
    ma, mb = np.meshgrid(x, gp['b'])
    arg = ma - mb
    variances = np.meshgrid(x, gp['sigma']*gp['sigma'])[1]
    amplitudes = np.meshgrid(x, gp['a'])[1]
    wa = amplitudes * np.exp(-0.5 * arg*arg / variances)
    if verbose:
        print "shapes: x", x.shape, "variances", variances.shape, "wa", wa.shape
        print "wa:", wa.min(), wa.mean(), wa.max(), wa.std()
    weights = wa.sum(axis=0) / wa.sum()

    if False:
        # 
        b = w+w+1
    
        m = np.zeros((b,), dtype=np.float)
        m[w] = 1.0
        corr_1d = scipy.ndimage.correlate1d(m, weights)
    
        #print "weights len=%d" % weights.shape, weights
        #print weights
        #print "corr_1d len=%d" % corr_1d.shape, corr_1d
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(x, weights, "r+")
        ax2.plot(range(b), corr_1d, "g+")
        plt.show()
        
        m = np.zeros((b, b), dtype=np.float)
        m[w,...] = corr_1d

    m = np.zeros((b,b), dtype=np.float)
    m[b//2,...] = weights
    ortho_weights = gaussian_filter1d_weights(ortho_sigma)
    wm = scipy.ndimage.correlate1d(m, ortho_weights, axis=0)

    if verbose:
        print "b", b, "weights.len", len(weights)
        print "wm shape", wm.shape
        print "wm:", wm.min(), wm.mean(), wm.max(), wm.std()

    if plot_unrot_kernel:
        plt.imshow(wm)
        plt.show()

    rotated = scipy.ndimage.rotate(wm, angle)
    row, col = divmod(rotated.argmax(), rotated.shape[0])
    if verbose:
        print "max at", row, col

    if plot_rot_kernel:
        plt.imshow(rotated)
        plt.show()

    if verbose:
        print "rotated sums:", rotated.sum(axis=0), rotated.sum(axis=1)
        print "rotated: shape", rotated.shape, "sum", rotated.sum()
        print "rotated:", rotated.min(), rotated.mean(), rotated.max(), rotated.std()
    return rotated
    
def write_aniso_kernel(m, sigma1, sigma2, angle):
    w = m.shape[0]
    write_kernel("aniso-%dx%d_%g_%g_%g.kern" % (w, w, sigma1, sigma2, angle))

def write_kernel(m, fname, comment=None):
    w = m.shape[0]
    with open(fname, "w") as fd:
        if comment:
            fd.write(comment+'\n')
        fd.write("np.array([")
        for i in range(w):
            fd.write('[')
            for j in range(w):
                fd.write("%g, " % m[i, j])
            fd.write('],\n')
        fd.write('])')

def parse_gauss_parm_str(s):
    data = StringIO(s.replace(':', '\n'))
    return np.genfromtxt(data, delimiter=',', names=["a", "b", "sigma"], dtype=(float,float,float))
    

if __name__ == '__main__':
    import sys
    import getopt

    def help():
        print """
%s - generate anisotropic rotated kernels

Use:
%s [options] <width> <sigma1> <sigma2> <angle>

Options:

-v     verbose
-h     help
-w <sigma> display weights for Gaussian filter with sigma <sigma>
-k <width,sigma,origin> display 1-D kernel
-p <a1,b1,sigma1:a2,b2,sigma2:...>
-r <angle>  rotation (with -p option only)
-n     do not write output file
""" % (progname, progname)

    np.set_printoptions(linewidth=1000, precision=3)

    progname = sys.argv[0]

    verbose = False
    do_write_output = True
    weight_sigma = None
    offset = None
    w_s_o = None
    gp = None
    angle = None
    

    opts, args = getopt.getopt(sys.argv[1:], "hvnw:W:k:p:r:")
    for opt, value in opts:
        if opt == '-h':
            help()
            sys.exit(0)
        elif opt == '-v':
            verbose = True
        elif opt == '-n':
            do_write_output = False
        elif opt == '-w':
            weight_sigma = float(value)
        elif opt == '-W':
            weight_sigma, offset = [float(s) for s in value.split(',')]
        elif opt == '-k':
            w_s_o = [float(s) for s in value.split(',')]
            w_s_o[0] = int(w_s_o[0])
            w_s_o[2] = int(w_s_o[2])
        elif opt == '-p':
            gp = parse_gauss_parm_str(value)
        elif opt == '-r':
            angle = float(value)

    if gp is not None:
        print gp
        #generate_sum_gauss(int(weight_sigma) if weight_sigma else 11, gp, 1.1, angle if angle else 0.0)
        psf = generate_sum_gauss(gp, 1.1, angle if angle else 0.0)
        U, s, V = np.linalg.svd(psf, full_matrices=True, compute_uv=True)
        print "results of Singular Value Decomposition of %dx%d PSF matrix" % psf.shape
        print U; print s; print V
        write_kernel(psf, "psf.kern")
        sys.exit(0)

    if offset is not None:
        print gaussian_filter1d_weights_offset(weight_sigma, offset)
        sys.exit(0)

    if weight_sigma is not None:
        print gaussian_filter1d_weights(weight_sigma)
        sys.exit(0)

    if w_s_o is not None:
        print gaussian_filter1d_origin(*w_s_o)
        sys.exit(0)

    try:
        #w, sigma1, sigma2, angle = [float(s) for s in sys.argv[1:]]
        w, sigma1, sigma2, angle = [float(s) for s in args]
    except ValueError:
        help()
        sys.exit(1)

    w = int(w)

    m = generate_aniso_gauss_kernel(w, (sigma1, sigma2), angle)
    if verbose:
        print m, m.sum()
    
    if do_write_output:
        write_aniso_kernel(m, sigma1, sigma2, angle)
