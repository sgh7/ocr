#!/usr/bin/python

def bc(i):
    mask = 0x80
    c = 0
    while mask:
        if i&mask:
            c += 1
        mask >>= 1
    return c

with open("bitcounts.h", "w") as fd:
    fd.write("unsigned char bcounts[256] = {")
    for i in range(256):
        if i%16 == 0:
            fd.write("\n\t")
        c = bc(i)
        fd.write("%d, " % c)
    fd.write("\n};\n")
        
