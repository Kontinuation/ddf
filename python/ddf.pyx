from cython.operator cimport dereference as deref
from libcpp.string cimport string
import numpy as np
cimport numpy as np
cimport cython


class ddf_exception(Exception):
    """
    Common exception type used through out ddf
    """
    pass


cdef extern from "../include/nd_array.hh" namespace "ddf":
    cdef cppclass vector[T]:
        vector(int len) except +
        int size()
        void resize(int new_size)
        string to_string()
        char *c_str()
        void copy_from(T *buf)
        int shape(int k)
        T &operator [](int)
        T *raw_data()


cdef class vector_float64:
    cdef vector[double] *v
    def __init__(self, np.ndarray[np.float64_t, ndim=1, mode="c"] nparr =
                 np.ndarray(0, dtype=np.dtype(float))):
        self.v = new vector[double](0)
        self.from_array(nparr)

    def __dealloc__(self):
        del self.v
        self.v = NULL

    def from_array(self, np.ndarray[np.float64_t, ndim=1, mode="c"] nparr):
        self.__memcheck()
        self.v.resize(nparr.shape[0])
        if nparr.shape[0] > 0:
            self.v.copy_from(&nparr[0])
        return self

    def to_array(self):
        self.__memcheck()
        cdef int arrlen = self.v.size()
        nparr = np.ndarray(arrlen, dtype=np.dtype(float))
        for i in range(arrlen): nparr[i] = deref(self.v)[i]
        return nparr

    def __str__(self):
        self.__memcheck()
        return self.v.c_str()

    def __memcheck(self):
        if self.v is NULL:
            raise ddf_exception("vector not initialized")


cdef extern from "../include/expr.hh" namespace "ddf":
    cdef cppclass math_expr[T]:
        math_expr(int type) except +
        string to_string()
        void eval(vector[T] &y)
        vector[T] delta

    cdef cppclass constant[T]:
        constant(vector[T] &v) except +
        void eval(vector[T] &y)
        vector[T] _v


def __test__():
    v = vector_float64(np.array([1.0, 2.0, 3.0, 4.0]))
    a = new constant[double](deref(v.v))
    b = vector_float64()
    a.eval(deref(b.v))
    print str(b)
    pass

