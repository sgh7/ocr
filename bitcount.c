/*
 * bitcount.c  --  count bits set in a Python object
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "bitcounts.h"

typedef unsigned int uint;
typedef unsigned char uchar;

/* From: Adam Zalcman's answer to 
   http://stackoverflow.com/questions/9829578/fast-way-of-counting-non-zero-bits-in-python
 */

static int CountBits(uint n) {
	n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1);
	n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2);
	n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4);
	n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8);
	n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16);
	n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32);
	return n;
}



static int
bcount(const uchar *buf, Py_ssize_t size) {
	int c = 0;

	if ( buf != NULL ) {
		for ( ; size >= sizeof(uint); size -= sizeof(uint)) {
			c += CountBits(*((uint *)buf) );
			buf += sizeof(uint);
		}
		while (size-- > 0 )
			c += bcounts[*buf++];
	}
	return c;
}



static PyObject *
bitcount_bitcount(PyObject *self, PyObject *args) {
	int rc = 0;
	Py_ssize_t size;
	const uchar *buf;

	if ( !PyArg_ParseTuple(args, "s#", &buf, &size) )
		return NULL;
	rc = bcount(buf, size);
	return Py_BuildValue("i", rc);
}

static PyMethodDef BitcountMethods[] = {
    {"bitcount",  bitcount_bitcount, METH_VARARGS,
     "Count the number of bits set in an array.."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initbitcount(void)
{
    (void) Py_InitModule("bitcount", BitcountMethods);
}

