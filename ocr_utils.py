#!/usr/bin/python

class Pickled(object):
    pass

def show_glyph(bool_matrix):
    w, h = bool_matrix.shape
    print w, h, np.bincount(bool_matrix.ravel())
    for j in range(h):
        print ''.join(['*' if b_value else ' ' for b_value in bool_matrix[j,::]])


