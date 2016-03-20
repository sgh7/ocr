#!/usr/bin/python
# coding: utf-8

import sys
import getopt
import numpy as np
from skimage import io
from skimage import filters
from skimage import img_as_ubyte
import skimage.segmentation as seg
from scipy import ndimage
from scipy import signal
from skimage import measure
from skimage.filters.rank import median, enhance_contrast
from skimage.morphology import disk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cPickle
from ocr_utils import *
from valley import find_local_minimum
from scipy import interpolate

def help():
    print """
%s select features from a single page image

Use:
 %s [options] <img-file>

options:

-h         this help
-v         be verbose
-k <file>  read specification of kernel to convolve with incoming image from file
-c [R|G|B] select color plane from incoming image
-d <delta> amount to tweak threshold value from Otsu algorithm
-m         fold pixels to monochrome by averaging RGB values
-M <x,y>   specify maximum glyph sizes in pixels
-s         generate splines for segmentation assistance
-o <outfile>  output pickled results

   x
""" % (progname, progname)

def plot_with_histogram(img, title='', Otsu_threshold=None, used_threshold=None):
    #img_bytes = img_as_ubyte(img, force_copy=True)
    hist = np.histogram(img, bins=256)
    max_ordinate = np.max(hist[0])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title(title)
    ax2.plot(hist[1][:-1], hist[0], lw=2)
    if Otsu_threshold:
        ax2.plot([Otsu_threshold]*2, [0, max_ordinate], 'g-')
    if used_threshold:
        ax2.plot([used_threshold]*2, [0, max_ordinate], 'r-')
    ax2.set_title('Histogram of grey values')
    plt.show()

class Contour(object):
    def __init__(self, cont):
        self.cont = cont
        self.cent = None

    def centroid(self):
        if not self.cent:
            self.cent = self.cont[::, 0].mean(), self.cont[::, 1].mean()
        return self.cent

    def size(self):
        return self.cont[::, 0].max() - self.cont[::, 0].min(),\
               self.cont[::, 1].max() - self.cont[::, 1].min(),\

    def npoints(self):
        return self.cont.shape[0]

    def is_closed(self):
        # FIXME: make this more succinct
        return self.cont[0, 0] == self.cont[-1, 0] and \
               self.cont[0, 1] == self.cont[-1, 1]

def glyph_copy(mask, labeled_glyphs, label, size_y, size_x, gly_min_y, gly_min_x, h, w):
    """copy glyph from original image.

    mask    original image after modified Otsu algorithm applied
    labeled_glyphs result of labelling original image
    label           label to be copied
    size_y, size_x  dimensions of output bitmap - should be same for all
                    invocations
    gly_min_y, gly_min_x  offsets within original image
    h, w            height, width of bounding box of labeled region"""

    out = np.zeros((size_y, size_x), dtype=np.bool)
    for i in range(h):
        for j in range(w):
            if labeled_glyphs[gly_min_y+i, gly_min_x+j] == label:
                out[i, j] = mask[gly_min_y+i, gly_min_x+j]
    print label, size_y, size_x, gly_min_y, gly_min_x, h, w
    show_glyph(out)
    return out
    
def show_image(label, img, show_values=False):
    print "%s image is %d-D of shape %s" % (label, img.ndim, img.shape),
    print "dtype=%s min=%f max=%f" % (img.dtype.name, img.ravel().min(), img.ravel().max())
    if show_values:
        print img



progname = sys.argv[0]
verbose = False
kern_file = None
fold_pixels_to_monochrome = False
colour_plane = None
otsu_tweak = 13
max_bb_x = 48
max_bb_y = 48
outfname = None
splines = None

vslice_samp_window = 150
threshold_arg_curvature = 0.003
depth_threshold = 0.3
width_threshold = 0.2

try:
    (opts, files) = getopt.getopt(sys.argv[1:], "hvmc:M:o:k:t:d:s")
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
    elif flag == '-M':
        max_bb_x, max_bb_y = tuple([int(s) for s in value.split(',')])
    elif flag == '-c':
        colour_plane = value
    elif flag == '-o':
        outfname = value
    elif flag == '-d':
        otsu_tweak = float(value)
    elif flag == '-k':
        kern_file = value
    elif flag == '-s':
        splines = []
    else:
        print >>sys.stderr, "%s: unknown flag %s" % (progname, flag)
        sys.exit(5)

fname = files[0]

img = io.imread(fname)

show_image("Original", img)

sample_how = None
if fold_pixels_to_monochrome:
    img = (img[::,::,0]+img[::,::,1]+img[::,::,2])/3
    sample_how = "Averaged"
elif colour_plane:
    img = img[::,::,"RGB".index(colour_plane)] + 0.0
    sample_how = "Colour "+colour_plane

if sample_how:
    show_image(sample_how, img)
else:
    sample_how = "Original"

plt.title(sample_how+" image")
plt.imshow(img, cmap = cm.Greys_r)
plt.show()

def deriv2_of_horiz_bands(img, hband_size):
    """For each horizontal band in image, calculate second
    derivatives of pixel densities.


    Divide the image into horizontal bands with each 
    band consisting of hband_size scanlines.

    For each band, sum the pixels in the vertical direction.

    Return the second difference of the summation in the
    corresponding result row, as well as the detrended
    sums divided by the standard deviation of the sums
    for each band.
    """

    #FIXME: a partial band at the bottom of the image is ignored
    img_height, img_width = img.shape[:2]
    n_bands = img_height / hband_size
    d2 = np.empty((n_bands, img_width-2))
    detr = np.empty((n_bands, img_width))
    for sl_index in range(n_bands):
        window_top = sl_index*hband_size
        band = img[window_top:window_top+hband_size,...]
        sum_ = band.sum(axis=0)
        d2[sl_index,...] = np.diff(sum_, n=2)
        dd = -signal.detrend(sum_)
        detr[sl_index,...] = dd / dd.std()
    return np.pad(d2, ((0,0), (1,1)), 'edge'), detr



def plot_band0(img, vslice_samp_window):
    img_width = img.shape[1]
    vsw0 = img[0:vslice_samp_window,...]
    sum0 = vsw0.sum(axis=0)
    var0 = vsw0.var(axis=0)
    dif0 = np.diff(sum0, n=2)
    fig, axes = plt.subplots(2,1, sharex=True)
    axes[0].set_title("Image Band 0")
    axes[0].imshow(vsw0)
    axes[0].set_xlim([0, img_width])
    axes[0].set_ylim([vslice_samp_window, 0])
    #axes[0].colorbar()
    sum_line, = axes[1].plot(np.arange(img_width), sum0, "k-", label="sum")
    var_line, = axes[1].plot(np.arange(img_width), var0, "r-", label="variance")
    dif0 = deriv2_of_horiz_bands(img, vslice_samp_window)[0][0,]
    dif_line, = axes[1].plot(np.arange(img_width), dif0, "g-", label="$2nd$ difference")
    plt.legend(handles=[sum_line, var_line, dif_line], loc="center right")
    axes[1].set_xlim([0, img_width])
    plt.show()
    print axes.shape
    print vsw0.shape


def plot_second_derivatives(detrended, dif, sel_x, sel_y, sel_z, vslice_samp_window):
    def closure(sel_x, sel_y, sel_z, dif):
        def format_coord(x, y):
            xi, yi = int(x), int(y)
            try:
                s = "%d" % dif[yi, xi]
            except IndexError:
                s = "?"
            return "x=%1.1f    y=%1.1f %s" % (x, y, s)
        return format_coord
    
    format_coord = closure(sel_x, sel_y, sel_z, dif)
    
    fig, ax = plt.subplots()
    ax.set_title("second-derivatives of mean pixel densities / band")
    ax.invert_yaxis()
    ax.scatter(sel_y, sel_x, c=sel_z, s=np.abs(sel_z), alpha=0.5)
    ax.format_coord = format_coord
    plt.xlabel("x")
    plt.ylabel("band $(w=%d)$" % vslice_samp_window)
    x = np.arange(detrended.shape[1])
    for i in range(detrended.shape[0]):
        ax.plot(x, -detrended[i,...] / 10.0 + i + 0.5, "g-")
    plt.show()


def walk_bands(band_origin, x, detrended):
    """Trace skew in thin vertical gap in text through sample bands.

    band_origin is which band the x value is from.

    x is the position in the originating band of the greatest (negative sign)
    curvature in the summed pixel intensities (saddle).

    detrended is a 2-D array of detrended sums for each band.

    Returns
    -------

    List of (maximum-depth-offset, width, depth) tuples in each
    band that are transitively nearest neighbours to the valley
    at (band_origin, x).
    """
    bands = [None] * detrended.shape[0]
    bands[band_origin] = x
    cx = x
    for i in range(band_origin, -1, -1):
        bands[i] = (m, w, d) = find_local_minimum(cx, detrended[i])
        cx = m

    cx = x
    for i in range(band_origin+1, len(bands)):
        bands[i] = (m, w, d) = find_local_minimum(cx, detrended[i])
        cx = m

    return bands

    
def fmt_neighbourhood(a, left, right):
    return ' '.join(["%.2f" % item for item in a[left:right+1]])


def gen_splines(img, vslice_samp_window, threshold_arg_curvature, depth_threshold, width_threshold):
    plot_band0(img, vslice_samp_window)
    
    dif, detrended = deriv2_of_horiz_bands(img, vslice_samp_window)
    zero_vals = np.zeros_like(dif)
    sel = np.where(dif < np.percentile(dif, 100*threshold_arg_curvature, axis=1, keepdims=True), dif, zero_vals)
    sel_x, sel_y = sel.nonzero()
    sel_z = dif[sel.nonzero()]
    
    plot_second_derivatives(detrended, dif, sel_x, sel_y, sel_z, vslice_samp_window)
    
    m_w_d = np.array([find_local_minimum(y, detrended[x]) for (x, y) in zip(sel_x, sel_y)])
    
    max_depth = m_w_d[...,2].max()
    
    width_pcentile = np.percentile(m_w_d[m_w_d[...,2]>max_depth*depth_threshold,1], 100*width_threshold)
    
    print m_w_d[m_w_d[...,2]>max_depth*depth_threshold,...]
    
    plt.title("distribution of valley widths and depths")
    plt.xlabel("widths")
    plt.ylabel("depths")
    plt.hist2d(m_w_d[...,1], m_w_d[...,2], bins=40)
    plt.show()
    
    band_points = []
    splines = []
    for i, (x, y, z) in enumerate(zip(sel_x, sel_y, sel_z)):
        (m, w, d) = m_w_d[i,...]
        if d > max_depth*depth_threshold and w < width_pcentile:
            print i, x,y,z, m,w,d, fmt_neighbourhood(detrended[x], m-w, m+w),
            if (m,w,d) in band_points:
                print "duplicate"
            else:
                bands = walk_bands(x, y, detrended)
                print bands
                band_points += bands
                s_x = np.arange(detrended.shape[0])*vslice_samp_window + vslice_samp_window/2.0
                s_y = [b[0] for b in bands]
                spl = interpolate.splrep(s_x, s_y, s=0.5)
                splines.append(spl)
    return splines

if splines is not None:
    splines = gen_splines(img, vslice_samp_window, threshold_arg_curvature, depth_threshold, width_threshold)
    print splines
    
kernel_3x3 = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])
highpass = ndimage.convolve(img, kernel_3x3)
# not actually using the convolved image...

if kern_file:
    with open(kern_file) as f:
        k_spec = [line for line in f.readlines() if not line.startswith('#')]
    kern = eval(''.join(k_spec))
    img = ndimage.convolve(img, kern)
    process_how = "Convolve with " + kern_file
else:
    gb_sigma = 15
    img = img - ndimage.gaussian_filter(img, gb_sigma)
    #process_how = "Gaussian Sharpening σ=%f" % gb_sigma
    process_how = "Gaussian Sharpening $\\sigma=%.1f$" % gb_sigma

show_image(process_how, img)
#plt.title(process_how+" image")
#plt.imshow(img, cmap = cm.Greys_r)
#plt.show()

#img = median(img/256.0, disk(1))
#img = enhance_contrast(img/256.0, disk(1))


# The Otsu method is a simple heuristic to find a 
# threshold to separate the foreground from background.
val = filters.threshold_otsu(img)
print "threshold from Otsu method is %f" % val
used_threshold = val+otsu_tweak
#FIXME: parameterize or estimate the literal constant in the previous line!!!
print "using threshold of %f" % used_threshold
mask = img < used_threshold  # fudge the threshold to minimize the number of connected regions

#plot_with_histogram(img, "Sharpened image", Otsu_threshold=val, used_threshold=used_threshold)
print process_how
plot_with_histogram(img, process_how, Otsu_threshold=val, used_threshold=used_threshold)

if False:
    for incr in np.linspace(8, 18, 11):
        mask_incr = img < (val+incr)
        plt.imshow(mask_incr, cmap = cm.Greys_r)
        contours = measure.find_contours(mask_incr, 0.5, fully_connected='high')
        plt.title("threshold=Glbl Otsu + %d, nc=%d" % (incr, len(contours)))
        plt.show()
#plot_with_histogram(mask)

mask_fat = mask.copy()
mask_fat[1::,::] |= mask[:-1:,::]
mask_fat[::,1::] |= mask[::,:-1:]
mask_fat[2::,::] |= mask[:-2:,::]
mask_fat[::,2::] |= mask[::,:-2:]

print "splines", splines
if splines is not None:
    y = range(img.shape[0])
    axis = plt.gca()
    for spline in splines:
        print "interpolating spline", spline
        x = interpolate.splev(y, spline, der=0).astype(int)
        mask[y,x] = False
        mask_fat[y,x] = False
        axis.plot(x, y, "r-")

plt.title("Fattened Image mask from modified Otsu")
plt.imshow(mask_fat, cmap='gray')
#plt.imshow(gauss_highpass, cmap = cm.Greys_r)
plt.show()

# Find contours at a constant value of 0.5
contours = measure.find_contours(mask, 0.5, fully_connected='high')
print "found %d contours" % len(contours)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(mask_fat, interpolation='nearest', cmap=plt.cm.gray)


contour_a = [None]*len(contours)
print "n centroid size #points is-closed"
for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    c = Contour(contour)
    contour_a[n] = c
    print n, c.centroid(), c.size(), c.npoints(), c.is_closed()
    #print n, contour, contour[:, 1], contour[:, 0]

ax.set_title("Fattened Image mask with Contours")
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()


segmentation = ndimage.binary_fill_holes(mask)
print "segmentation: shape %s type %s" % (segmentation.shape, segmentation.dtype.name)
labeled_glyphs, n_glyphs = ndimage.label(segmentation)
print "labeled glyphs: N=%d shape=%s type=%s" % (n_glyphs, labeled_glyphs.shape, labeled_glyphs.dtype.name)

plt.title("labeled glyphs")
plt.imshow(labeled_glyphs, cmap='gray')
plt.show()

n_labels = n_glyphs + 1
gly_min_x = [sys.maxint]*n_labels
gly_max_x = [-sys.maxint]*n_labels
gly_min_y = [sys.maxint]*n_labels
gly_max_y = [-sys.maxint]*n_labels


# determine bounding boxes for each label
for y in range(labeled_glyphs.shape[0]):
    for x in range(labeled_glyphs.shape[1]):
        label = labeled_glyphs[y, x]
        if x < gly_min_x[label]:
            gly_min_x[label] = x
        if x > gly_max_x[label]:
            gly_max_x[label] = x
        if y < gly_min_y[label]:
            gly_min_y[label] = y
        if y > gly_max_y[label]:
            gly_max_y[label] = y

glyphs = [None]*n_labels

fig, ax = plt.subplots()
ax.set_title("Contrast (unfattened) image with glyph bounding boxes")
ax.imshow(mask, interpolation='nearest', cmap=plt.cm.gray)

gly_size_hist = np.zeros((max_bb_x+2, max_bb_y+2), dtype='int32')

for label in range(1, n_labels):  # label 0 is the entire image background
    min_y, max_y = gly_min_y[label], gly_max_y[label]
    min_x, max_x = gly_min_x[label], gly_max_x[label]
    w_h = w = max_x+1-min_x
    h_h = h = max_y+1-min_y
    if w_h > max_bb_x:
        w_h = max_bb_x+1
    if h_h > max_bb_y:
        h_h = max_bb_y+1
    gly_size_hist[h_h, w_h] += 1
    if w <= max_bb_x and h <= max_bb_y:
        glyphs[label] = glyph_copy(mask, labeled_glyphs, label, max_bb_y+1, max_bb_x+1, min_y, min_x, h, w)

    print label, min_y, min_x, h, w
    #ax.plot(max_x, max_y, "r+")
    ax.plot([min_x, max_x], [min_y]*2, "r-")
    ax.plot([min_x, max_x], [max_y]*2, "r-")
    ax.plot([min_x]*2, [min_y, max_y], "r-")
    ax.plot([max_x]*2, [min_y, max_y], "r-")

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

column_labels = ["%d" % i for i in range(max_bb_x+1)]+[">"]
row_labels = ["%d" % i for i in range(max_bb_y+1)]+[">"]
fig, ax = plt.subplots()
ax.set_title("glyph sizes")
ax.set_xlabel("widths")
ax.set_ylabel("heights")
heatmap = ax.pcolor(gly_size_hist, cmap=plt.cm.Blues)
ax.set_xticks(np.arange(gly_size_hist.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(gly_size_hist.shape[1])+0.5, minor=False)
#ax.invert_yaxis()
#ax.xaxis.tick_top()
ax.set_xticklabels(row_labels, minor=False, rotation=90)
ax.set_yticklabels(column_labels, minor=False)
##plt.colorbar()
plt.show()


if outfname:
    o = Pickled()
    o.gly_min_x = gly_min_x
    o.gly_max_x = gly_max_x
    o.gly_min_y = gly_min_y
    o.gly_max_y = gly_max_y
    o.glyphs = glyphs[1:]     # ignore background
    with open(outfname, "w") as of:
       cPickle.dump(o, of)

# s = seg.felzenszwalb(mask, scale=0.03, sigma=1.0, min_size=3)
# print s.ravel().max()
# plt.imshow(s, cmap='gray')
# plt.show()
