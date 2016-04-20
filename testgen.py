#!/usr/bin/python

"""Generate test data."""

import numpy as np

N = 64
i = np.arange(N)

FDS = 27458 + i
latitude = np.rad2deg(np.arcsin(np.random.uniform(low=-1.0, high=1.0, size=N)))
longitude = 1.5*i + np.random.normal(16.0, 5.0)
slant_range = 10.0*(i-24)**2 -300.0*i + 15000.0
incident_angle = np.random.normal(60.0, 10.0, size=N)
emission_angle = np.random.normal(45.0, 15.0, size=N)
phase_angle = np.random.normal(110.0, 7.5, size=N)
frame_dim_hor = slant_range / 80.0
frame_dim_ver = frame_dim_hor * 0.75
resolution = slant_range / 50.0

print "{:>8s}{:>8s}{:>8s}{:>9s}{:>7s}{:>7s}{:>7s}{:>6s}{:>6s}{:>6s}" \
    .format(*"FDS lat long slant incid emiss phase dim-h dim-v res".split())
for j in i:
    print "{:8d}{:8.1f}{:8.1f}{:9,.0f}{:7.0f}{:7.0f}{:7.0f}{:6.0f}{:6.0f}{:6.0f}" \
          .format(FDS[j], latitude[j], longitude[j], slant_range[j],
                  incident_angle[j], emission_angle[j], phase_angle[j],
                  frame_dim_hor[j], frame_dim_ver[j], resolution[j])

