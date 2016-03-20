#!/usr/bin/python
# -*- coding: utf-8 -*-


def seek_bottom_in_direction(ix, a, direction):
    """Scan monotonically decreasing slope until bottom reached.

    ix is index to start from
    a is array of heights
    direction is one of [+1, -1]
    """
    curr_depth = a[ix]
    new_pos = ix+direction
    try:
        while True:
            if a[new_pos] <= curr_depth:
                curr_depth = a[new_pos]
                new_pos += direction
            else:
                new_pos -= direction
                break
    except IndexError:
        new_pos -= 1
    if new_pos < 0:
        new_pos = 0
    #print "seek_bottom_in_direction(%d, ..., %d) -> %d" % (ix, direction, new_pos)
    return new_pos

def seek_top_in_direction(ix, a, direction):
    """Scan monotonically increasing slope until peak reached.

    ix is index to start from
    a is array of heights
    direction is one of [+1, -1]
    """
    curr_height = a[ix]
    new_pos = ix+direction
    try:
        while True:
            if a[new_pos] >= curr_height:
                curr_height = a[new_pos]
                new_pos += direction
            else:
                new_pos -= direction
                break
    except IndexError:
        new_pos -= 1
    if new_pos < 0:
        new_pos = 0
    #print "seek_top_in_direction(%d, ..., %d) -> %d" % (ix, direction, new_pos)
    return new_pos


def find_local_minimum(ix, a):
    """Characterize valley in a at position ix.

    The valley consists of two opposed monotonically decreasing/increasing
    slopes about the deepest point.

    a is array of heights
    ix is the index into a containing the valley of interest.

    Returns tuple of:
    m index of deepest point in the valley
    w width of valley
    d depth of valley measured from lowest hilltop
    """

    curr_height = a[ix]

    # tell if we are in flat-land
    left_ix = right_ix = ix
    while left_ix > 0 and a[left_ix-1] == curr_height:
        left_ix -= 1
    right_max = len(a)-1
    while right_ix < right_max and a[right_ix+1] == curr_height:
        right_ix += 1
  
    if left_ix > 0:
        if a[left_ix-1] < curr_height:
            bottom = seek_bottom_in_direction(left_ix-1, a, -1)
            peak_left = seek_top_in_direction(bottom, a, -1)
            peak_right = seek_top_in_direction(left_ix, a, 1)
        else:
            peak_left = seek_top_in_direction(left_ix-1, a, -1)
            bottom = seek_bottom_in_direction(right_ix, a, 1)
            peak_right = seek_top_in_direction(bottom, a, 1)
    else:
        if a[right_ix+1] < curr_height:
            bottom = seek_bottom_in_direction(right_ix+1, a, 1)
            peak_left = 0
            peak_right = seek_top_in_direction(bottom, a, 1)
        else:
            peak_right = seek_top_in_direction(right_ix+1, a, 1)
            peak_left = 0
            bottom = right_ix

    height_left = a[peak_left]
    height_right = a[peak_right]
            
    result = bottom, peak_right-peak_left, min(height_left, height_right)-a[bottom]
    if False:
        print "find_local_minimum(%d)" % ix,
        if left_ix != ix or ix != right_ix:
            print "ix -> %d..%d" % (left_ix, right_ix),
        print "[surrounding peaks %d,%d] -> %s" % (peak_left, peak_right, result)
    return result

if __name__ == '__main__':
    import sys
    
    a = [0, 0, 0, 6, 12, 18, 16, 12, 5, -4, 3, 7, 11, 8, 9, 2, 1, 1, 7, 18, -5, 22, 7, 3]
    for arg in sys.argv[1:]:
        if arg == '-v':
            print a
        else:
            ix = int(arg)
            bottom, width, depth = find_local_minimum(ix, a)
            print "%d at ofs %d -> bottom=%d width=%d depth=%d" % (a[ix], ix, bottom, width, depth)
    
