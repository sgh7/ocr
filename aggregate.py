#!/usr/bin/python

import cPickle
import sys
from math import log
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import spatial

from ocr_utils import *

def get_np_gly_pos_dims(pin):
    arrays = [np.array(a) for a in [pin.gly_min_x, pin.gly_max_x, pin.gly_min_y, pin.gly_max_y, pin.gly_bits_set]]
    return tuple(arrays+[pin.img_shape, pin.glyphs])
    #return pin.gly_min_x, pin.gly_max_x, pin.gly_min_y, pin.gly_max_y, pin.glyphs

def get_xywh(i):
    x = gly_min_x[i]
    y = gly_min_y[i]
    w = 1+gly_max_x[i]-x
    h = 1+gly_max_y[i]-y
    return x, y, w, h

fname = sys.argv[1]
try:
    imgfname = sys.argv[2]
except IndexError:
    imgfname = None
vspace_hat = 53  # FIXME!!!!
hspace_hat = 30  # FIXME!!!!

with open(fname) as fd:
    pin = cPickle.load(fd)
    print "fudging img_shape for now..."
    pin.img_shape = (1920, 2560)
    print "in %s, I found %d unlabeled glyphs" % (fname, len(pin.glyphs))
    gly_min_x, gly_max_x, gly_min_y, gly_max_y, gly_bits_set, img_shape, glyphs = get_np_gly_pos_dims(pin)

print glyphs[0].shape

# compute centroids for each glyph
gly_mean_x = (gly_min_x + gly_max_x) / 2
gly_mean_y = (gly_min_y + gly_max_y) / 2

points = zip(gly_mean_y, gly_mean_x)

tree = spatial.cKDTree(points)

max_distance = 50  # pixels between features

distances, indices =  tree.query([1000,1000], k=20, distance_upper_bound=max_distance)

for i, (d, ix) in enumerate(zip(distances, indices)):
    if d > max_distance:
        break
    print i, d, ix, tree.data[ix], gly_min_y[ix], gly_min_x[ix]

def lookup(tree, y, x, from_, max_distance):
    distances, indices = tree.query([y, x], k=2, distance_upper_bound=max_distance)
    for (d, ix) in zip(distances, indices):
        if ix == from_:
            continue
        if d > max_distance:
            break
        return ix
    return None

#
# as the glyphs are scanned from top to bottom, 
# accum is updated from each pixel of the glyph

#d=float(sys.float_info.max_exp*0.8) / img_shape[0]
#multipliers = 2**(d*np.arange(img_shape[0]))

binary_exponent_range_len = sys.float_info.max_exp-sys.float_info.min_exp
if binary_exponent_range_len <= img_shape[0]:
    print "Image has %d scanlines but this machine only has %d float exponents."
    sys.exit(3)

multipliers = 2**np.linspace(sys.float_info.min_exp+2, sys.float_info.max_exp-3, img_shape[0])
accum = np.zeros((img_shape[1],), dtype=np.double)
d = multipliers[0] / multipliers[1]

print d
print multipliers[-10:]

def update_accum(accum, multipliers, glyph, y, x, h, w):
    mbase = multipliers[y:]
    abase = accum[x:]
    for j in range(h):
        for i in range(w):
            if glyph[j, i]:
                try:
                    #print "%g + %g" % (abase[i], mbase[j]),
                    abase[i] += mbase[j]
                    #print "is %g" % abase[i]
                except RuntimeWarning:
                    print "RTW:", j, i, j+y, i+x, abase[i], mbase[j]

def fmt_above(above):
    return '['+','.join(["%.3g" % f for f in above])+']'

def log_distance(l1, l2):
    try:
        return log(l1/l2) - log(d)
    except ZeroDivisionError:
        return 123456789

def same_column(tree, left, right, weld=False):
    """Tell if two glyphs are in the same textual column.

    Use heuristics from the accumulators.
    """
    left_y, left_x = tree.data[left]
    right_y, right_x = tree.data[right]
    try:
        min_accum = accum[left_x:right_x].min()
    except ValueError:
        min_accum = 0
    higher_glyph_y = min(gly_min_y[left], gly_min_y[right])
    ldist = log_distance(multipliers[higher_glyph_y], min_accum)
    is_same = ldist < vspace_hat*1.5
    print "same_column?", tree.data[left], tree.data[right], min_accum, higher_glyph_y, ldist, is_same
    if is_same and weld:
        lower_glyph_y = max(gly_max_y[left], gly_max_y[right])
        addend = multipliers[lower_glyph_y]
        for x in range(int(left_x), int(right_x)):
            accum[x] += addend
    return is_same

pairings = {}

def add_pair(a, b, how):
    a, b = (min(a,b), max(a, b))
    pairings[(a,b)] = how

for i in range(1, len(glyphs)):
    x, y, w, h = get_xywh(i)
    my_level = multipliers[y]
    above = accum[x:x+w]
    amax = above.max()
    ldist = log_distance(my_level, amax)
    print i, y, x, h, w, gly_bits_set[i], my_level, amax, ldist, fmt_above(above)
    update_accum(accum, multipliers, glyphs[i], y, x, h, w)
    other = None
    how = None
    if ldist < 10:  # FIXME!!
        # found a glyph above this one
        other = lookup(tree, y-ldist, x, i, vspace_hat/2)
        how = "v"
    else:
        left = lookup(tree, y, x-hspace_hat/2, i, hspace_hat/2)
        if left and same_column(tree, left, i, weld=True):
            other = left
            how = "h"
    if other:
        print "joining %d with %d (%s)" % (i, other, how)
        add_pair(i, other, how)
            
        
if imgfname:
    img = img = io.imread(imgfname)
    plt.title("image with aggregations")
    plt.imshow(img, cmap = cm.Greys_r)
    ax = plt.gca()
    print pairings
    td = tree.data
    for (k, how) in pairings.iteritems():
        print k, how
        a, b = k
        style = "r-" if how == 'h' else "g-"
        ax.plot([td[a][1], td[b][1]], [td[a][0], td[b][0]], style)
    plt.show()
    
