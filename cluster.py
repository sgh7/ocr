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
from ocr_utils import *


def compose_cl_view(glyphs, clusters, labels, width, margins_tblr, hs, vs):
    """Create image file showing results of clustering.

    glyphs       N x (w x h) array of incoming glyphs 
    clusters     per-glyph is the assigned cluster
    labels       N-element list containing labels or None
    width        width in pixels of the resulting image
    margins_tblr tuple of (top,bottom,left,right) margins
    hs           horizontal spacing between glyphs
    vs           vertical spacing between glyphs
    """

    counts = Counter()
    for cl in clusters:
        counts[cl] += 1
    cl_by_size = counts.most_common(None)

    # image interior width
    iw = width - margins_tblr[2] - margins_tblr[3]
    lmarg = margins_tblr[2]
    tmarg = margins_tblr[0]

    # glyph width and height
    gw = glyphs[0].shape[0]
    gh = glyphs[0].shape[1]

    ###gc = [None]*len(glyphs)    # which cluster each glyph assigned to

    def advance(x, y):
        x += gw+hs
        if x >= iw:
            x = hs
            y += gh+vs
        return (x, y)
        
    # pre-allocate positions of glyphs within clusters
    # ranked by descending cluster size
    cl_render_positions = [None]*(len(cl_by_size)+1)
    red_markers = [None]*len(cl_by_size)
    y = vs
    x = hs
    for i, (cl, count) in enumerate(cl_by_size):
        cl_rp = [None]*count
        for j in range(count):
            cl_rp[j] = (x,y)
            x, y = advance(x, y)
        x, y = advance(x, y)   
        red_markers[i] = (x,y)
        x, y = advance(x, y)
        cl_render_positions[cl] = cl_rp
            
    height = y+vs+gh+margins_tblr[0] + margins_tblr[1]
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # fill the image

    # first the glyphs, via the clusters
    cl_used = [0]*(1+len(cl_by_size))   # indexes through each cluster
    for glyph_index, cl in enumerate(clusters):
        # for each glyph, which cluster (origin-1 indexing!!) it's in
        try:
            (x, y) = cl_render_positions[cl][cl_used[cl]]
        except IndexError:
            print "*ouch(%d)*" % cl
            continue
        x += lmarg
        y += tmarg
        cl_used[cl] += 1
        gl = glyphs[glyph_index]
        if gl is None:
            continue
        if labels[glyph_index] is None:
            colors = [0,1,2]
        else:
            colors = [2]      # labeled glyphs rendered blue
        print "gli %d in cl %d  at (%d,%d) %s" % (glyph_index, cl, y, x, "blue" if labels[glyph_index] else "white")
        for i in range(gw):
            for j in range(gh):
                try:
                    img[y+j, x+i, colors] = gl[j,i]*128
                except IndexError:
                    print "*yikes(%d,%d)*" % (y+j, x+i)
                except ValueError:
                    print "missing glyph at %d" % (glyph_index)
            

    # now the red lines separating the clusters
    for rm in red_markers:
        (x,y) = rm
        x += lmarg
        y += tmarg
        for i in range(gw/2-1, gw/2+1):
            for j in range(gh):
                try:
                    img[y+j, x+i, 0] = 128
                except IndexError:
                    print "*yikes(%d,%d)*" % (y+j, x+i)
    return img


def help():
    print """
%s glyph clustering and training for OCR

Use:
 %s [options] <pickle-files>

options:

-h         this help
-v         be verbose
-s <input-similarities file>
-c <method> specify method to calculate similarity matrix [1..%d]
-S <output-similarities file>
-t <input-training data file>
-T <output-training data file>
-m <method> specify method for determining distances between clusters
-r <threshold>  specify clustering threshold
-w <cluster-show-file>   write character glyphs to this image file
                         ordered by cluster

   Maintain training data.  Glyphs from pickle files are merged
with any existing training data files and clustered.  The program
then selects up to a constant number of samples from each cluster
including any existing training data and prompts the user to
classify any glyphs that are not yet within the training data,
which is then output.

""" % (progname, progname, CALC_MAX)

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

progname = sys.argv[0]
verbose = False
in_train_files = []
out_train_file = None
in_sim_file = None
out_sim_file = None
cl_method = "single"
cl_threshold = 0.3
w_cluster_file = None

# clip bounding boxes by making these small
train_width = 1000
train_height = 1000

max_labeled_per_cluster = 2

sim_calc = CALC_PACKBITS
#sim_calc = CALC_SCIPY

try:
    (opts, files) = getopt.getopt(sys.argv[1:], "hvt:T:s:S:m:r:w:c:")
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
    elif flag == '-s':
        sim_calc = CALC_FROMFILE
        in_sim_file = value
    elif flag == '-c':
        sim_calc = int(value)
    elif flag == '-S':
        out_sim_file = value
    elif flag == '-m':
        validate_cl_method(value)
        cl_method = value
    elif flag == '-r':
        cl_threshold = float(value)
    elif flag == '-w':
        w_cluster_file = value
    else:
        print >>sys.stderr, "%s: unknown flag %s" % (progname, flag)
        sys.exit(5)

if in_sim_file and len(in_train_files) > 0:
    print >>sys.stderr, "cannot use -s with -t"
    sys.exit(4)

glyphs = []
for fname in files:
    with open(fname) as fd:
        pin = cPickle.load(fd)
        glyphs += pin.glyphs
        # FIXME: check if the glyph sizes conform with those from previous files
        print "in %s, I found %d unlabeled glyphs" % (fname, len(pin.glyphs))

new_count = len(glyphs)
labels = [None]*new_count
        
for fname in in_train_files:
    with open(fname) as fd:
        pin = cPickle.load(fd)
        glyphs += pin.glyphs
        # FIXME: check if the glyph sizes conform with those from previous files
        labels += pin.labels
        print "in %s, I found %d glyphs and %d labels" % (fname, len(pin.glyphs), len(pin.labels))

print "now have %d glyphs and %d labels" % (len(glyphs), len(labels))

if False:
    show_glyph(glyphs[1])
    show_glyph(glyphs[2])
    show_glyph(glyphs[1] ^ glyphs[2])
    print jaccard_sim(glyphs[1], glyphs[2])

n = len(glyphs)
p = glyphs[0].size

sim = calc_similarity_matrix(glyphs, sim_calc, n, p, in_sim_file)

print sim

verify_square(sim)

if out_sim_file:
    o = Pickled()
    o.sim = sim
    with open(out_sim_file, "w") as of:
       cPickle.dump(o, of)



plt.subplot(121)
plt.imshow(sim, interpolation="nearest")

dissimilarity = distance.squareform(1-sim)
linkage = hierarchy.linkage(dissimilarity, method=cl_method)
clusters = hierarchy.fcluster(linkage, cl_threshold, criterion="distance")
print "linkage", len(linkage), linkage
print "clusters", len(clusters), clusters
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

def class_help():
    print "classification: press Enter to skip an item, HELP for help,"
    print "    Ctl-D or NEXT to quit cluster, type QUIT to quit classifying"

if out_train_file:
    np_clusters = np.array(clusters)
    gl_indices = np.arange(n)
    cl_by_size = counts.most_common(None)
    class_help()
    we_are_done = False
    for cl, nglyphs in cl_by_size:
        contained_glyphs = gl_indices[np_clusters == cl]
        if contained_glyphs.size != nglyphs:
            print "*nglyphs(%d) != contained_glyphs.size(%d)*" % (nglyphs, contained_glyphs.size)
        num_labeled = count_labeled_glyphs(contained_glyphs)
        print "cluster %d has %d labels, %d samples" % (cl, num_labeled, nglyphs),
        print contained_glyphs.tolist()
        if num_labeled >= max_labeled_per_cluster:
            continue
        for pm in np.random.permutation(nglyphs):
            gli = contained_glyphs[pm]
            if labels[gli]:
                print "%d already labeled as %s" % (gli, labels[gli])
                continue
            elif glyphs[gli] is None:
                print "undefined glyph %d, skipping cluster %d" % (gli, cl)
                continue
            else:
                print gli
            show_glyph(glyphs[gli])
            try:
                label = raw_input("above is: ")
                if label == 'HELP':
                    class_help()
                    label = raw_input("above is: ")
                if label == 'QUIT':
                    we_are_done = True
                    break
                elif len(label) == 0:
                    continue
                elif label == 'NEXT':
                    break
                labels[gli] = label
                num_labeled += 1
                if num_labeled >= max_labeled_per_cluster:
                    break
            except EOFError:
                break
        if we_are_done:
            break



if w_cluster_file:
    print "composing cluster view image"
    img = compose_cl_view(glyphs, clusters, labels, width=800, 
                          margins_tblr=(40,40,40,40), hs=8, vs=8)
    print "writing cluster view image to file", w_cluster_file
    io.imsave(w_cluster_file, img)
    print "written cluster view image"


plt.subplot(122)
hierarchy.dendrogram(linkage, orientation='right', color_threshold=0.3)
plt.xlabel("Event number")
plt.ylabel("Dissimilarity")
plt.show()

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

