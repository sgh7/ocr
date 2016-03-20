#!/usr/bin/python
# -*- coding: utf-8 -*-

from valley import *

class test_V(object):

    @classmethod
    def setup_class(cls):
        cls.a = [0, 0, 0, 6, 12, 18,           # valley 0   ..5
                   16, 12, 5, -4, 3, 7, 11,    # valley 1  5..12
                   8, 9,                       # valley 2 12..14
                   2, 1, 1, 7, 18,             # valley 3 14..19
                   -5, 22,                     # valley 4 19..21
                   7, 3]                       # valley 5 21..

    @classmethod
    def teardown_class(cls):
        pass

    def test_seek_bottom_0_left(self):
        assert seek_bottom_in_direction(0, self.a, -1) == 0

    def test_seek_bottom_0_right(self):
        assert seek_bottom_in_direction(0, self.a, 1) == 2

    def test_seek_bottom_5_left(self):
        assert seek_bottom_in_direction(5, self.a, -1) == 0

    def test_seek_bottom_5_right(self):
        assert seek_bottom_in_direction(5, self.a, 1) == 9

    def test_seek_top_0_left(self):
        assert seek_top_in_direction(0, self.a, -1) == 0

    def test_seek_top_0_right(self):
        assert seek_top_in_direction(0, self.a, 1) == 5

    def test_seek_top_5_left(self):
        assert seek_top_in_direction(5, self.a, -1) == 5

    def test_seek_top_5_right(self):
        assert seek_top_in_direction(5, self.a, 1) == 5

    def test_seek_top_6_left(self):
        assert seek_top_in_direction(6, self.a, -1) == 5

    def test_seek_top_6_right(self):
        assert seek_top_in_direction(6, self.a, 1) == 6

    def test_seek_top_13_left(self):
        assert seek_top_in_direction(13, self.a, -1) == 12

    def test_seek_top_13_right(self):
        assert seek_top_in_direction(13, self.a, 1) == 14

    def test_valley0(self):
        assert find_local_minimum(0, self.a)[2] == 0

    def test_valley1(self):
        assert find_local_minimum(7, self.a) == (9, 7, 15)

    def test_valley2(self):
        assert find_local_minimum(13, self.a) == (13, 2, 1)

    def test_valley_boundary_2_3(self):
        bottom_width_depth = find_local_minimum(14, self.a) 
        assert bottom_width_depth == (13, 2, 1) or bottom_width_depth == (16, 5, 8)

    def test_valley3(self):
        assert find_local_minimum(15, self.a) == (17, 5, 8)

    def test_valley4(self):
        assert find_local_minimum(20, self.a) == (20, 2, 23)

    def test_valley5(self):
        assert find_local_minimum(22, self.a)[2] == 0
