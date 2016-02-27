#!/usr/bin/python

import sys
import os
import getopt
import numpy as np
from skimage import io
from skimage import filters
from skimage import img_as_ubyte
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cPickle
import time
import bitstring
from bitcount import bitcount

CALC_NONE  = 0
CALC_PACKBITS = 1
CALC_BITSTRING = 2
CALC_NUMPY = 3

def help():
    print """
%s glyph clustering and training for OCR

Use:
 %s [options] <pickle-files>

options:

-h         this help
-v         be verbose
-t <input-training data file>
-T <output-training data file>

   Maintain training data.  Glyphs from pickle files are merged
with any existing training data files and clustered.  The program
then selects up to a constant number of samples from each cluster
including any existing training data and prompts the user to
classify any glyphs that are not yet within the training data,
which is then output.

""" % (progname, progname)

class Pickled(object):
    pass

class TrainDataItem(object):
    def __init__(self, glyph, label=None):
        self.glyph = glyph
        self.label = label


def plot_with_histogram(img, title=''):
    print img.dtype.name, img.ravel().min(), img.ravel().max()
    #img_bytes = img_as_ubyte(img, force_copy=True)
    hist = np.histogram(img, bins=256)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title(title)
    ax2.plot(hist[1][:-1], hist[0], lw=2)
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

def show_glyph(bool_matrix):
    w, h = bool_matrix.shape
    print w, h, np.bincount(bool_matrix.ravel())
    for j in range(h):
        print ''.join(['*' if b_value else ' ' for b_value in bool_matrix[j,::]])

def true_count(m):
    """Count number of True values in a boolean array.

    numpy.bincount on a boolean array returns a singleton result iff
    all values in the argument are False."""

    try:
        c = np.bincount(m.ravel())[1]
    except IndexError:
        c = 0
    return c

def jaccard_sim(m1, m2, verbose=False):
    """Compute Jaccard similarity of two NumPy boolean arrays."""

    #intersect_count = np.bincount((m1 & m2).ravel())[1]
    #union_count = np.bincount((m1 | m2).ravel())[1]
    intersect_count = true_count(m1 & m2)
    union_count = true_count(m1 | m2)
    if verbose:
        print "js: %d/%d" % (intersect_count, union_count)
    try:
        jc = intersect_count / float(union_count)
    except ZeroDivisionError:
        jc = 1.0
    return jc
    
def jaccard_sim_numpy_packed(m1, m2, verbose=False):
    """Compute Jaccard similarity of two NumPy packed arrays."""

    intersect_count = bitcount(m1 & m2)
    union_count = bitcount(m1 | m2)
    if verbose:
        print "js: %d/%d" % (intersect_count, union_count)
    try:
        jc = intersect_count / float(union_count)
    except ZeroDivisionError:
        jc = 1.0
    return jc
    
def jaccard_sim_bitstring(m1, m2, verbose=False):
    """Compute Jaccard similarity of two bitstring boolean arrays."""

    intersect_count = (m1 & m2).count(True)
    union_count = (m1 | m2).count(True)
    if verbose:
        print "js: %d/%d" % (intersect_count, union_count)
    try:
        jc = intersect_count / float(union_count)
    except ZeroDivisionError:
        jc = 1.0
    return jc
    

def glyph_copy(mask, labeled_glyphs, label, size_x, size_y, gly_min_x, gly_min_y, w, h):
    """copy glyph from original image.

    mask    original image after modified Otsu algorithm applied
    labeled_glyphs result of labelling original image
    label           label to be copied
    size_x, size_y  dimensions of output bitmap - should be same for all
                    invocations
    gly_min_x, gly_min_y  offsets within original image
    w, h            width, height of bounding box of labeled region"""

    out = np.zeros((size_x, size_y), dtype=np.bool)
    for i in range(w):
        for j in range(h):
            if labeled_glyphs[gly_min_x+i, gly_min_y+j] == label:
                out[i, j] = mask[gly_min_x+i, gly_min_y+j]
    print label, size_x, size_y, gly_min_x, gly_min_y, w, h
    show_glyph(out)
    return out
    

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

progname = sys.argv[0]
verbose = False
in_train_files = []
out_train_file = None

# clip bounding boxes by making these small
train_width = 1000
train_height = 1000

sim_calc = CALC_PACKBITS

try:
    (opts, files) = getopt.getopt(sys.argv[1:], "hvt:T:")
except getopt.GetoptError, exc:
    print >>sys.stderr, "%s: %s" % (progname, str(exc))
    sys.exit(1)

for flag, value in opts:
    if flag == '-h':
        help()
        sys.exit(1)
    elif flag == '-v':
        verbose = True
    elif flag == '-t':
        in_train_files.append(value)
    elif flag == '-T':
        out_train_file = value
    else:
        print >>sys.stderr, "%s: unknown flag %s" % (progname, flag)
        sys.exit(5)

glyphs = []
for fname in files:
    with open(fname) as fd:
        pin = cPickle.load(fd)
        print len(pin.glyphs)
        glyphs += pin.glyphs

new_count = len(glyphs)
labels = [None]*new_count
        
for fname in in_train_files:
    with open(fname) as fd:
        pin = cPickle.load(fd)
        glyphs += pin.glyphs
        labels += pin.labels

show_glyph(glyphs[1])
show_glyph(glyphs[2])
show_glyph(glyphs[1] ^ glyphs[2])
print jaccard_sim(glyphs[1], glyphs[2])

n = len(glyphs)
p = glyphs[0].size

if sim_calc == CALC_PACKBITS:
    # optimize by packing the bits with numpy.packbits
    cast_to_uint8 = np.array([0], dtype=np.uint8)
    dummy_ba = np.packbits(np.array([True]*p) + cast_to_uint8)
    b_glyphs = [None]*n
    print "converting to bitstrings using numpy.packbits",
    time_start = time.time()
    for i in range(n):
        try:
            ba = np.packbits(glyphs[i] + cast_to_uint8)
        except TypeError:
            ba = dummy_ba
        b_glyphs[i] = ba
        print '.',
    print
    print "converted to bitstrings in %f seconds" % (time.time()-time_start)
    print "computing similarity matrix with bitcount",
    time_start = time.time()
    sim = np.zeros((n,n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i+1, n):
            try:
                s = jaccard_sim_numpy_packed(b_glyphs[i], b_glyphs[j], verbose=(0,1315)==(i,j))
            except TypeError:
                # a glyph exceeded the bounding box and therefore
                # was probably a non-textual item such as a line
                s = 0.0
            sim[i, j] = s
            sim[j, i] = s
        print '.',
    print
    
    print "computed (bitstring) similarity matrix in %f seconds" % (time.time()-time_start)
    print sim
    
elif sim_calc == CALC_BITSTRING:
    # optimize by packing the bits with bitstring
    dummy_ba = [True]*p
    b_glyphs = [None]*n
    fmt = "%d*bool" % p
    print "converting to bitstrings",
    time_start = time.time()
    for i in range(n):
        try:
            ba = glyphs[i].ravel().tolist()
        except AttributeError:
            ba = dummy_ba
        b_glyphs[i] = bitstring.pack(fmt, *ba)
        print '.',
    print
    print "converted to bitstrings in %f seconds" % (time.time()-time_start)
    print "computing similarity matrix with bitstring",
    time_start = time.time()
    sim = np.zeros((n,n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i+1, n):
            try:
                s = jaccard_sim_bitstring(b_glyphs[i], b_glyphs[j], verbose=(0,1315)==(i,j))
            except TypeError:
                # a glyph exceeded the bounding box and therefore
                # was probably a non-textual item such as a line
                s = 0.0
            sim[i, j] = s
            sim[j, i] = s
        print '.',
    print
    
    print "computed (bitstring) similarity matrix in %f seconds" % (time.time()-time_start)
    print sim
    
elif sim_calc == CALC_NUMPY:
    print "computing similarity matrix with numpy",
    time_start = time.time()
    sim = np.zeros((n,n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i+1, n):
            try:
                s = jaccard_sim(glyphs[i], glyphs[j], verbose=(0,1315)==(i,j))
            except TypeError:
                # a glyph exceeded the bounding box and therefore
                # was probably a non-textual item such as a line
                s = 0.0
            sim[i, j] = s
            sim[j, i] = s
        print ".",
    print
    
    print "computed (numpy) similarity matrix in %f seconds" % (time.time()-time_start)
    print sim
else:
    print >>sys.stderr, "sim_calc method undefined."
    sys.exit(2)

if out_train_file:
    pout = Pickled()
    nglyphs = np.array(glyphs)
    nlabels = np.array(labels)
    indices = [i for i in range(len(labels)) if labels[i] != None]
    print "indices", indices
    pout.glyphs = nglyphs[indices,...].tolist()
    pout.labels = nlabels[indices].tolist()
    with open(out_train_file, "w") as fd:
        cPickle.dump(pout, fd)

sys.exit(0)
fname = files[0]

img = io.imread(fname)

if fold_pixels_to_monochrome:
    img = (img[::,::,0]+img[::,::,1]+img[::,::,2])/3
elif colour_plane:
    img = img[::,::,"RGB".index(colour_plane)] + 0.0
print img
print img.ndim
print img.shape

kernel_3x3 = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])
highpass = ndimage.convolve(img, kernel_3x3)

lowpass = ndimage.gaussian_filter(img, 15)
gauss_highpass = img - lowpass

img = gauss_highpass

plot_with_histogram(img, "high-passed image")

#img = median(img/256.0, disk(1))
#img = enhance_contrast(img/256.0, disk(1))


# The Otsu method is a simple heuristic to find a 
# threshold to separate the foreground from background.
val = filters.threshold_otsu(img)
print val
mask = img < val+13  # fudge the threshold to minimize the number of connected regions

if False:
    for incr in np.linspace(8, 18, 11):
        mask_incr = img < (val+incr)
        plt.imshow(mask_incr, cmap = cm.Greys_r)
        contours = measure.find_contours(mask_incr, 0.5, fully_connected='high')
        plt.title("threshold=Glbl Otsu + %d, nc=%d" % (incr, len(contours)))
        plt.show()
#plot_with_histogram(mask)

plt.imshow(img, cmap = cm.Greys_r)
plt.show()

mask_fat = mask.copy()
mask_fat[1::,::] |= mask[:-1:,::]
mask_fat[::,1::] |= mask[::,:-1:]
mask_fat[2::,::] |= mask[:-2:,::]
mask_fat[::,2::] |= mask[::,:-2:]

plt.imshow(mask_fat, cmap='gray')
#plt.imshow(gauss_highpass, cmap = cm.Greys_r)
plt.show()

# Find contours at a constant value of 0.5
contours = measure.find_contours(mask, 0.5, fully_connected='high')
print len(contours)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(mask_fat, interpolation='nearest', cmap=plt.cm.gray)



contour_a = [None]*len(contours)
for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    c = Contour(contour)
    contour_a[n] = c
    print n, c.centroid(), c.size(), c.npoints(), c.is_closed()
    #print n, contour, contour[:, 1], contour[:, 0]

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()


segmentation = ndimage.binary_fill_holes(mask)
print segmentation.shape, segmentation.dtype.name
labeled_glyphs, n_glyphs = ndimage.label(segmentation)
print n_glyphs, labeled_glyphs.shape, labeled_glyphs.dtype.name

plt.imshow(labeled_glyphs, cmap='gray')
plt.show()

n_labels = n_glyphs + 1
gly_min_x = [sys.maxint]*n_labels
gly_max_x = [-sys.maxint]*n_labels
gly_min_y = [sys.maxint]*n_labels
gly_max_y = [-sys.maxint]*n_labels


# determine bounding boxes for each label
for x in range(labeled_glyphs.shape[0]):
    for y in range(labeled_glyphs.shape[1]):
        pixel = labeled_glyphs[x, y]
        if x < gly_min_x[pixel]:
            gly_min_x[pixel] = x
        if x > gly_max_x[pixel]:
            gly_max_x[pixel] = x
        if y < gly_min_y[pixel]:
            gly_min_y[pixel] = y
        if y > gly_max_y[pixel]:
            gly_max_y[pixel] = y

glyphs = [None]*n_labels

fig, ax = plt.subplots()
ax.imshow(mask_fat, interpolation='nearest', cmap=plt.cm.gray)

gly_size_hist = np.zeros((max_bb_x+2, max_bb_y+2), dtype='int32')

for label in range(1, n_labels):  # label 0 is the entire image background
    w_h = w = gly_max_x[label]+1-gly_min_x[label]
    h_h = h = gly_max_y[label]+1-gly_min_y[label]
    if w_h > max_bb_x:
        w_h = max_bb_x+1
    if h_h > max_bb_y:
        h_h = max_bb_y+1
    gly_size_hist[w_h, h_h] += 1
    if w <= max_bb_x and h <= max_bb_y:
        glyphs[label] = glyph_copy(mask, labeled_glyphs, label, max_bb_x+1, max_bb_y+1, gly_min_x[label], gly_min_y[label], w, h)

    print label, gly_min_x[label], gly_min_y[label], w, h
    ax.plot(gly_max_y[label], gly_max_x[label], "r+")

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

column_labels = ["%d" % i for i in range(max_bb_x+1)]+[">"]
row_labels = ["%d" % i for i in range(max_bb_y+1)]+[">"]
fig, ax = plt.subplots()
heatmap = ax.pcolor(gly_size_hist, cmap=plt.cm.Blues)
ax.set_xticks(np.arange(gly_size_hist.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(gly_size_hist.shape[1])+0.5, minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.set_xticklabels(row_labels, minor=False, rotation=90)
ax.set_yticklabels(column_labels, minor=False)
plt.show()


if outfname:
    class Pickled(object):
        pass
    o = Pickled()
    o.gly_min_x = gly_min_x
    o.gly_max_x = gly_max_x
    o.gly_min_y = gly_min_y
    o.gly_max_y = gly_max_y
    o.glyphs = glyphs
    with open(outfname, "w") as of:
       cPickle.dump(o, of)

# s = seg.felzenszwalb(mask, scale=0.03, sigma=1.0, min_size=3)
# print s.ravel().max()
# plt.imshow(s, cmap='gray')
# plt.show()
