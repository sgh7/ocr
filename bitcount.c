/*
 * bitcount.c  --  count bits set in a Python object
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "bitcounts.h"

static int
bcount(const unsigned char *buf, Py_ssize_t size) {
	int c = 0;

	if ( buf != NULL ) {
		while (size-- > 0 )
			c += bcounts[*buf++];
	}
	return c;
}



static PyObject *
bitcount_bitcount(PyObject *self, PyObject *args) {
	int rc = 0;
	Py_ssize_t size;
	const unsigned char *buf;

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

