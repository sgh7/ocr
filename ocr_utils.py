#!/usr/bin/python

import numpy as np
import time
from bitcount import bitcount
import bitstring
from scipy.spatial import distance

class Pickled(object):
    pass

def show_glyph(bool_matrix):
    w, h = bool_matrix.shape
    print w, h, np.bincount(bool_matrix.ravel())
    for j in range(h):
        print ''.join(['*' if b_value else ' ' for b_value in bool_matrix[j,::]])

# clustering-related

CALC_NONE  = 0
CALC_PACKBITS = 1
CALC_BITSTRING = 2
CALC_NUMPY = 3
CALC_SCIPY = 4
CALC_FROMFILE = 5
CALC_MAX = 5


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
    
def calc_similarity_matrix(glyphs, sim_calc, n, p, in_sim_file):
    """Compute similarity matrix.

    glyphs -- 
    sim_calc -- method index
    n  --
    p --
    in_sim_file -- name of file containing matrix to be read in and returned
    """
    #FIXME ... above docstring
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
    elif sim_calc == CALC_SCIPY:
        print "computing similarity matrix with scipy",
        time_start = time.time()
        ba_dummy = np.ones(glyphs[0].shape, dtype=np.bool)
        f_glyphs = np.zeros((n, p), dtype=np.bool)
        for i in range(n):
            try:
                f_glyphs[i,:] = glyphs[i].ravel()
            except AttributeError:
                f_glyphs[i,:] = ba_dummy.ravel()
        print '.',
        dissimilarity = distance.pdist(f_glyphs, 'jaccard')
        print '.',
        sim = 1 - distance.squareform(dissimilarity)
        print "computed (scipy) similarity matrix in %f seconds" % (time.time()-time_start)
    elif sim_calc == CALC_FROMFILE:
        with open(in_sim_file) as f:
            o = cPickle.load(f)
            sim = o.sim
    else:
        print >>sys.stderr, "sim_calc method %d undefined." % sim_calc
        sys.exit(2)
    return sim

def verify_square(m):
    if len(m.shape) != 2:
        print >>sys.stderr, "bad shape", m.shape
    if m.shape[0] != m.shape[1]:
        print >>sys.stderr, "bad shape", m.shape
    for i in range(m.shape[0]):
        for j in range(i+1, m.shape[1]):
            if m[i,j] != m[j,i]:
                print >>sys.stderr, "content mis-match at (%d,%d): %f <> %f" % (i,j,m[i,j],m[j,i])


def validate_cl_method(m):
    valid_methods = ['single', 'complete', 'average', 'weighted', 
                     'centroid', 'median', 'ward']
    #FIXME: can only use the 1st four
    if m in valid_methods:
        return
    print >>sys.stderr, "clustering method %s unknown" % m
    print >>sys.stderr, "valid methods are: ", ' '.join(valid_methods)
    sys.exit(3)

