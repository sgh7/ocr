#!/usr/bin/python

import sys
from image5 import five
from skimage import io
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import getopt
from skimage import color, data, restoration
from aniso_kern import parse_gauss_parm_str, generate_sum_gauss
import pymc as pm

def run_mcmc(gp, transverse_sigma=1.0, motion_angle=0.0):
    """Estimate PSF using Markov Chain Monte Carlo

    gp - Gaussian priors
    transverse_sigma - prior
    motion_angle - prior
    """

    motion_angle = pm.VonMises("motion_angle", motion_angle, 1.0)
    transverse_sigma = pm.Exponential("transverse_sigma", 1.0)
    N = gp.shape[0]

    mixing_coeffs = pm.Exponential("mixing_coeffs", 1.0, size=N)
    #mixing_coeffs.set_value(gp['a'])
    mixing_coeffs.value = gp['a']
    longitudinal_sigmas = pm.Exponential("longitudinal_sigmas", 1.0, size=N)
    #longitudinal_sigmas.set_value(gp['sigma'])
    longitudinal_sigmas.value = gp['sigma']
    longitudinal_means = pm.Normal("longitudinal_means", 0.0, 0.04, size=N)
    #longitudinal_means.set_value(gp['b'])
    longitudinal_means.value = gp['b']

    dtype=np.dtype([('a', np.float),('b', np.float),('sigma', np.float)])

    @pm.deterministic
    def psf(mixing_coeffs=mixing_coeffs, longitudinal_sigmas=longitudinal_sigmas, \
            longitudinal_means=longitudinal_means, transverse_sigma=transverse_sigma, motion_angle=motion_angle):
        gp = np.ones((N,), dtype=dtype)
        gp['a'] = mixing_coeffs
        gp['b'] = longitudinal_means
        gp['sigma'] = longitudinal_sigmas
        return generate_sum_gauss(gp, transverse_sigma, motion_angle)

    trial_psf = generate_sum_gauss(gp, 2.0, 50.0, plot_unrot_kernel=True, plot_rot_kernel=True, verbose=True)
    print "trial_psf", trial_psf.min(), trial_psf.mean(), trial_psf.max(), trial_psf.std()
    obs_psf = pm.Uniform("obs_psf", lower=-1.0, upper=1.0, doc="Point Spread Function", value=trial_psf, observed=True, verbose=False)
    
    
    mcmc = pm.MCMC([motion_angle, transverse_sigma, mixing_coeffs, longitudinal_sigmas, longitudinal_means, obs_psf])
    pm.graph.dag(mcmc, format='png')
    plt.show()
    mcmc.sample(20000, 1000)

    motion_angle_samples = mcmc.trace("motion_angle")[:]
    transverse_sigma_samples = mcmc.trace("transverse_sigma")[:]
    
    ax = plt.subplot(211)
    plt.hist(motion_angle_samples, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_\\theta$", color="#A60628", normed=True)
    plt.legend(loc="upper right")
    plt.title("Posterior distributions of $p_\\theta$, $p_\\sigma$")

    ax = plt.subplot(212)
    plt.hist(transverse_sigma_samples, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_\\sigma$", color="#467821", normed=True)
    plt.legend(loc="upper right")
    plt.show()

    print mcmc.stats()
    mcmc.write_csv("out.csv")
    pm.Matplot.plot(mcmc)
    plt.show()


    

def plot_image(img, title):
    figure, axes = plt.subplots(1, 2)
    axes[0].imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    axes[0].set_title(title)
    axes[1].hist(img.ravel(), bins=40)
    axes[1].set_title("histogram")
    plt.show()

def plot_images(arr):
    figure, axes = plt.subplots(len(arr), 2)
    for i, (img, title) in enumerate(arr):
        ii = i+i
        axes[i][0].imshow(img, interpolation='nearest', cmap=plt.cm.gray)
        axes[i][0].set_title(title)
        axes[i][1].hist(img.ravel(), bins=40)
        axes[i][1].set_title("histogram")
    plt.show()


def image_edge_pixels(img):
    w, h = img.shape
    return np.r_[img[...,0], img[-1,1:-1], img[...,-1], img[0,1:-1]]

def pre_pad_image(img, pre_pad):
    pad_value = int(image_edge_pixels(img).mean())
    return np.pad(img, pre_pad, mode='constant', constant_values=pad_value)


def help():
    print """
%s blur and deblur image with added noise

Use:
 %s [options] <img-file>

options:

-h         this help
-v         be verbose
-k <file>  read specification of kernel to convolve with incoming image from file
-c [R|G|B] select color plane from incoming image
-m         fold pixels to monochrome by averaging RGB values
-P <pixels> pad outside of true image before convolution

-p <a1,b1,sigma1:a2,b2,sigma2:...> specify Gaussian functions to be summed
-s <sigma> sigma for Gaussian blur in orthogonal direction
-a <motion_angle> rotation (with -p option only)

-T <img file> target (blurry) image file to compare result of blurring with
-N <nf>    add noise factor
-M         run MCMC
""" % (progname, progname)

progname = sys.argv[0]
verbose = False
kern_file = None
fold_pixels_to_monochrome = False
colour_plane = None
gp = None
transverse_sigma = None
motion_angle = None
noise_factor = None
pre_pad = None
compare_img = None
do_mcmc = False

try:
    (opts, files) = getopt.getopt(sys.argv[1:], "hvmc:k:s:p:a:N:P:T:M:")
except getopt.GetoptError, exc:
    print >>sys.stderr, "%s: %s" % (progname, str(exc))
    sys.exit(1)

for flag, value in opts:
    if flag == '-h':
        help()
        sys.exit(1)
    elif flag == '-v':
        verbose = True
    elif flag == '-m':
        fold_pixels_to_monochrome = True
    elif flag == '-c':
        colour_plane = value
    elif flag == '-k':
        kern_file = value
    elif flag == '-p':
        gp = parse_gauss_parm_str(value)
    elif flag == '-a':
        motion_angle = float(value)
    elif flag == '-s':
        transverse_sigma = float(value)
    elif flag == '-N':
        noise_factor = float(value)
    elif flag == '-P':
        pre_pad = int(value)
    elif flag == '-T':
        compare_img = io.imread(value)
    elif flag == '-M':
        do_mcmc = True
    else:
        print >>sys.stderr, "%s: unknown flag %s" % (progname, flag)
        sys.exit(5)

img_fname = files[0]

img = io.imread(img_fname)

sample_how = None
if fold_pixels_to_monochrome:
    img = img.mean(axis=2)
    sample_how = "Averaged"
elif colour_plane:
    img = img[::,::,"RGB".index(colour_plane)] + 0.0
    sample_how = "Colour "+colour_plane
else:
    sample_how = "Original"

if do_mcmc:
    run_mcmc(gp, transverse_sigma=transverse_sigma if transverse_sigma else 1.0, motion_angle=motion_angle if motion_angle else 0.0)
    sys.exit(0)

if pre_pad:
    img = pre_pad_image(img, pre_pad)
    if compare_img is not None:
        compare_img = pre_pad_image(compare_img, pre_pad)
        

plt.title(sample_how+" image")
plt.imshow(img, cmap = cm.Greys_r)
plt.show()

orig_img = img.copy()

kern = None

if kern_file:
    with open(kern_file) as f:
        k_spec = [line for line in f.readlines() if not line.startswith('#')]
    kern = eval(''.join(k_spec))
    process_how = "Convolve with " + kern_file
elif gp is not None:
    kern = generate_sum_gauss(gp, transverse_sigma if transverse_sigma else 1.0, motion_angle if motion_angle else 0.0)
    process_how = "Convolve with generated kernel"

if kern is not None:
    img = ndimage.convolve(img, kern)

if noise_factor is not None:
    img += noise_factor * img.std() * np.random.standard_normal(img.shape)

plt.title(process_how+" image")
plt.imshow(img, cmap = cm.Greys_r)
plt.show()

if kern is not None:
    print "img.shape", img.shape, "kern.shape", kern.shape
    if min(img.shape) < min(kern.shape):
        pad = max(kern.shape) - min(img.shape)
        #img = np.pad(img, pad, mode='reflect')
        #img = np.pad(img, pad, mode='edge')
        pixels = image_edge_pixels(img)
        print pixels.min(), pixels.mean(), pixels.max()
        pad_value = int(image_edge_pixels(img).mean())
        print pad, type(pad), pad_value, type(pad_value)
        img = np.pad(img, pad, mode='constant', constant_values=pad_value)
        plt.title("padding inserted")
        plt.imshow(img, cmap = cm.Greys_r)
        plt.show()
    if compare_img is not None:
        img_diff = img.astype(int)-compare_img
        print "sd of diff", img_diff.std()
        plot_images([(img, "Blurred Original"), (img_diff, "Image minus reference"), (compare_img, "Reference")])
    deconv, chain = restoration.unsupervised_wiener(img, kern, reg=None, user_params=None, is_real=True, clip=False)
    print chain
    plt.title("after unsupervised Wiener restoration")
    plt.imshow(deconv, cmap = cm.Greys_r)
    plt.show()


    plot_image(deconv-orig_img, "Image difference")

