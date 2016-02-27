all:	bitcount.so

bitcounts.h: genbitcounts.py
	python genbitcounts.py

bitcount.so:  bitcounts.h bitcount.c
	gcc -O2 -fPIC -I/usr/include/python2.7 -shared -lpython2.7 -o bitcount.so bitcount.c

clean:
	rm bitcounts.h bitcount.so
