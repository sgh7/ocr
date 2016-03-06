#!/usr/bin/python

import sys
import os
import getopt
import numpy as np
from collections import Counter
from skimage import io
from skimage import filters
from skimage import img_as_ubyte
from scipy import ndimage
from skimage import measure
from scipy.cluster import hierarchy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cPickle
import time
import bitstring
from bitcount import bitcount
from collections import defaultdict
from ocr_utils import *

def verify(condition, message):
    if not eval(condition, globals()):
        print >>sys.stderr, "%s: %s" % (message, condition)
        sys.exit(10)

def help():
    print """
%s classify character features in image 

Use:
 %s [options] <img-file>

options:

-h         this help
-v         be verbose
-f <feature file>  input file containing features (-o option to feature_sel.py)
-t <input-training data file>

   Classify text features using features and training data outputted by
the feature_sel.py and cluster.py programs.
""" % (progname, progname)

# read image file and determine dimensions of bitmap

# create graphic preview bitmap

# read training data  (error if none!)

# read feature file (must exist)

# classify the features against the training data
#   method a)
#      re-run hierarchical clustering on combined feature and training data
#      assign features according to cluster membership
#          if the cluster contains no items from any training set
#              the feature will get an un-assigned status
#
#   other methods?  SVM?

# for each feature
#   if it is sufficiently close to an item in the training set
#       blit the graphic from the training set onto the graphic preview bitmap
#            color it depending on how good a match it is
#   else
#       blit the original bitmap in a distinct color

# create text match array
# for each feature
#   if it is not noise (small number of pixels total)
#       append target string, x, y, goodness of fit measurement
#      

# slope-detection - determine which glyphs are on which line

# output text


def similarities_to_labeled_gylphs(glyph, gcl):
    dists = [None]*len(gcl)
    #show_glyph(glyph)
    for i, g in enumerate(gcl):
        #show_glyph(g)
        dists[i] = jaccard_sim(glyph, g)
    return dists

def get_xywh(i):
    x = gly_min_x[i]
    y = gly_min_y[i]
    w = 1+gly_max_x[i]-x
    h = 1+gly_max_y[i]-y
    return x, y, w, h

def get_gly_pos_dims(pin):
    #return pin.gly_min_x, pin.gly_max_x, pin.gly_min_y, pin.gly_max_y, pin.glyphs
    print "get_gly_pos_dims switcheroo!!"
    return pin.gly_min_y, pin.gly_max_y, pin.gly_min_x, pin.gly_max_x, pin.glyphs

def compose_resolved_img(height, width, pin, clusters, labeled_clusters, lcl, gcl):

    gly_min_x, gly_max_x, gly_min_y, gly_max_y, glyphs = get_gly_pos_dims(pin)
    # background of glyphs ignored
    glyphs = [np.zeros((glyphs[0].shape), dtype=np.bool)] + glyphs  # FIXME: doing this twice...

    img = np.zeros((height, width), dtype=np.uint8)

    print "#clusters %d #glyphs %d" % (len(clusters), len(glyphs))
    for i, cl in enumerate(clusters[1:], 1):
        gl = glyphs[i]
        if gl is None:
            if verbose:
                print "glyph %d omitted" % i
            continue
        x, y, w, h = get_xywh(i)
        if verbose:
            print "glyph %d in cluster %d" %(i, cl),
            print "at (%d,%d) size (%d,%d)" %(y,x,h,w),
        if cl in labeled_clusters:
            similarities = similarities_to_labeled_gylphs(gl, gcl[cl])
            if verbose:
                print "labeled", lcl[cl], "dists", similarities
            intensity = 54+max(similarities)*200
        else:
            if verbose:
                print "unlabeled"
            intensity = 30
        for ax1 in range(h):
            for ax2 in range(w):
                try:
                    img[y+ax1, x+ax2] = gl[ax1, ax2]*intensity
                except IndexError:
                    print "*ouch*"
    return img




sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

progname = sys.argv[0]
verbose = False
in_train_files = []
in_feature_file = None
cl_method = "single"
cl_threshold = 0.3

sim_calc = CALC_PACKBITS
#sim_calc = CALC_SCIPY

try:
    (opts, files) = getopt.getopt(sys.argv[1:], "hvt:T:m:r:w:f:")
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
    elif flag == '-m':
        validate_cl_method(value)
        cl_method = value
    elif flag == '-r':
        cl_threshold = float(value)
    elif flag == '-w':
        w_cluster_file = value
    elif flag == '-f':
        in_feature_file = value
    else:
        print >>sys.stderr, "%s: unknown flag %s" % (progname, flag)
        sys.exit(5)

glyphs = []
fname = files[0]

img = io.imread(fname)
if verbose:
    print img.shape, img.dtype



with open(in_feature_file) as fd:
    pin = cPickle.load(fd)
    gly_min_x, gly_max_x, gly_min_y, gly_max_y, glyphs = get_gly_pos_dims(pin)
    # background of glyphs ignored

if verbose:
    print glyphs[:2]
    print len(glyphs)
    print glyphs[0].shape
    for id in ["gly_min_x", "gly_max_x", "gly_min_y", "gly_max_y"]:
        v = eval(id)
        print id, len(v), v[:5]

verify("gly_min_x[0] == 0", "invalid state")
verify("gly_min_y[0] == 0", "invalid state")
verify("gly_max_x[0]+1 == img.shape[1]", "invalid state")
verify("gly_max_y[0]+1 == img.shape[0]", "invalid state")

hist_widths = [0]*(1+glyphs[0].shape[0])
hist_heights = [0]*(1+glyphs[0].shape[1])
for i in range(1, len(glyphs)):
    if glyphs[i] is None:
        continue
    x, y, w, h = get_xywh(i)
    try:
        hist_widths[w] += 1
        hist_heights[h] += 1
    except IndexError:
        print "*UGH*", i, w, h

bit_sums_x = np.sum(img, axis=(0,2))
bit_sums_y = np.sum(img, axis=(1,2))
rfft_x = np.fft.rfft(bit_sums_x)
rfft_y = np.fft.rfft(bit_sums_y)
pow_x = rfft_x**2
pow_y = rfft_y**2
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
ax2.plot(np.arange(1, len(bit_sums_x)), bit_sums_x[1:], 'bo')
ax1.plot(np.arange(1, len(bit_sums_y)), bit_sums_y[1:], 'ro')
ax4.plot(np.arange(1, len(pow_x)), np.abs(rfft_x)[1:], 'b-')
ax3.plot(np.arange(1, len(pow_y)), np.abs(rfft_y)[1:], 'r-')
ax6.plot(np.arange(len(hist_widths)), hist_widths, 'b*')
ax5.plot(np.arange(len(hist_heights)), hist_heights, 'r*')
plt.show()


# restore "dummy" glyph as placeholder for entire image
#glyphs = np.vstack([np.zeros((glyphs[0].shape), dtype=np.bool), glyphs])
print len(glyphs)
glyphs = [np.zeros((glyphs[0].shape), dtype=np.bool)] + glyphs
print len(glyphs)

new_count = len(glyphs)
labels = [None]*new_count
        
for fname in in_train_files:
    with open(fname) as fd:
        tin = cPickle.load(fd)
        glyphs += tin.glyphs
        # FIXME: check if the glyph sizes conform with those from previous files
        labels += tin.labels
        print "in %s, I found %d glyphs and %d labels" % (fname, len(tin.glyphs), len(tin.labels))
        print "labels found are:", tin.labels

print "now have %d glyphs and %d labels" % (len(glyphs), len(labels))

if True:
    show_glyph(glyphs[1])
    show_glyph(glyphs[2])
    show_glyph(glyphs[1] ^ glyphs[2])
    print jaccard_sim(glyphs[1], glyphs[2])

n = len(glyphs)
p = glyphs[0].size

sim = calc_similarity_matrix(glyphs, CALC_PACKBITS, n, p, None)

print sim

verify_square(sim)


#plt.subplot(121)
#plt.imshow(sim, interpolation="nearest")

dissimilarity = distance.squareform(1-sim)
linkage = hierarchy.linkage(dissimilarity, method=cl_method)
clusters = hierarchy.fcluster(linkage, cl_threshold, criterion="distance")
print "linkage", len(linkage), linkage
print "clusters", len(clusters), clusters
labeled_clusters = frozenset(clusters[new_count:])
print "labeled clusters are:", list(labeled_clusters)
counts = Counter()
for cl in clusters:
    counts[cl] += 1
print len(counts), "unique clusters"
print counts

def count_labeled_glyphs(contained_glyphs):
    label = None
    count = 0
    for gli in contained_glyphs:
        if labels[gli] is not None:
            count += 1
            if label is None:
                label = labels[gli]
            elif labels[gli] != label:
                print "*ERROR* cluster contains conflicting labels %s %s" % (label, labels[gli])
    return count

lcl = {}
gcl = defaultdict(list)
for i, cl in enumerate(clusters[new_count:], new_count):
    print "%d in %d %s" % (i, cl, labels[i])
    lcl[cl] = labels[i]
    gcl[cl].append(glyphs[i])

if verbose:
    raw_input("Press enter to continue.")

res_img = compose_resolved_img(img.shape[0], img.shape[1], pin, clusters[:new_count], labeled_clusters, lcl, gcl)
fig, ax = plt.subplots()
#ax.imshow(res_img, interpolation='nearest', cmap=plt.cm.bwr)
ax.imshow(res_img, interpolation='nearest', cmap=plt.cm.CMRmap)
plt.show()

#plt.title("Sums of pixel values")
bit_sums_x = np.sum(res_img, axis=0)
bit_sums_y = np.sum(res_img, axis=1)
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(np.arange(len(bit_sums_x)), bit_sums_x, 'bo')
ax2.plot(np.arange(len(bit_sums_y)), bit_sums_y, 'ro')
plt.show()

# read-out

class Glyph(object):
    def __init__(self, id, x, y, w, h):
        self.id = id
        self.x = x
        self.ymin = y
        self.w = w
        self.h = h
        self.ymax = y+h-1
        self.shadows = set()
        self.shadowedby = set()
        self.label = None
        self.outputted = False

    __slots__ = ("id", "x", "ymin", "ymax", "w", "h", "shadows", "shadowedby", "label", "outputted")

    def compare(self, g2):
        if self.ymin > g2.ymax or self.ymax < g2.ymin:
            pass  # no contact
        elif self.x < g2.x:
            self.shadows |= set([g2.id])
            g2.shadowedby |= set([self.id])
        else:
            self.shadowedby |= set([g2.id])
            g2.shadows |= set([self.id])
        
sys.setrecursionlimit(30)

def find_ga_index(ga, y_ofs, start_index, end_index):
    print "fgi: %d %d %d" % (y_ofs, start_index, end_index)
    if ga[end_index].ymin <= y_ofs:
        return end_index
    elif ga[start_index].ymin >= y_ofs:
        return start_index
    elif start_index == end_index:
        return start_index
    else:
        middle = (start_index+end_index)/2
        if middle == start_index:
            return start_index
        if ga[middle].ymin > y_ofs:
            return find_ga_index(ga, y_ofs, start_index, middle-1)
        else:
            return find_ga_index(ga, y_ofs, middle, end_index)

def find_in_row(ga, y_ofs, max_delta_y):
    # depend upon the glyph-array being sorted by y-coordinate
    # of upper-left corner
    ix1 = find_ga_index(ga, y_ofs, 0, new_count-1)
    ix2 = find_ga_index(ga, y_ofs+max_delta_y, 0, new_count-1)
    print "found %d+%d in %d..%d" % (y_ofs, max_delta_y, ix1, ix2)
    

ga = [None]*new_count
for i in range(new_count):
    y = 0
    #if glyphs[i] is None:
    if False:
        ga[i] = Glyph(i, 0, y, 0, 0)
        continue
    x, y, w, h = get_xywh(i)
    print y,
    ga[i] = Glyph(i, x, y, w, h)

print

y_ofs = 0
find_in_row(ga, y_ofs, glyphs[0].shape[0])



#plt.subplot(122)
#hierarchy.dendrogram(linkage, orientation='left', color_threshold=0.3)
#plt.xlabel("Event number")
#plt.ylabel("Dissimilarity")
#plt.show()
#
