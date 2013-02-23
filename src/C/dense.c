/*
 * Copyright 2004-2009 J. Dahl and L. Vandenberghe.
 *
 * This file is part of CVXOPT version 1.1.2
 *
 * CVXOPT is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * CVXOPT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#define BASE_MODULE

#include "Python.h"
#include "cvxopt.h"
#include "misc.h"

#include <complexobject.h>

/* NumPy array protocol */
typedef struct {
  int version;
  int nd;
  char typekind;
  int itemsize;
  int flags;
  int_t *shape;
  int_t *strides;
  void *data;
} PyArrayInterface;

static const char PY_ARRAY_TC[3] = { 'i', 'f', 'c' };

/* prototyping and forward declarations */
extern void (*axpy[])(int *, number *, void *, int *, void *, int *) ;
extern void (*scal[])(int *, number *, void *, int *) ;
extern void (*gemm[])(char *, char *, int *, int *, int *, void *, void *,
    int *, void *, int *, void *, void *, int *) ;
extern void (*mtx_abs[])(void *, void *, int) ;
extern int (*div_array[])(void *, number, int) ;
extern int (*mtx_rem[])(void *, number, int) ;

extern void (*write_num[])(void *, int, void *, int) ;
extern int (*convert_num[])(void *, void *, int, int_t) ;
extern PyObject * (*num2PyObject[])(void *, int) ;
int get_id(void *, int ) ;

extern const int  E_SIZE[];
extern const char TC_CHAR[][2];
extern number One[3], MinusOne[3], Zero[3];

extern PyObject *base_mod;

extern PyTypeObject spmatrix_tp ;

PyTypeObject matrix_tp ;
matrix * Matrix_NewFromNumber(int_t , int_t , int_t , void *, int ) ;
static PyNumberMethods matrix_as_number ;
static PyObject * matrix_iter(matrix *) ;


void dscal_(int *, double *, double *, int *) ;
void zscal_(int *, complex *, complex *, int *) ;
void daxpy_(int *, double *, double *, int *, double *, int *) ;
void zaxpy_(int *, complex *, complex *, int *, complex *, int *) ;
void dgemm_(char *, char *, int *, int *, int *, double *, double *,
    int *, double *, int *, double *, double *, int *) ;
void zgemm_(char *, char *, int *, int *, int *, complex *, complex *,
    int *, complex *, int *, complex *, complex *, int *) ;



static const char err_mtx_list2matrix[][35] =
    {"not an integer list",
        "not a floating point list",
        "not a complex floating point list" };

#define free_convert_mtx_alloc(O1, O2, id) { \
    if (MAT_BUF(O1) != O2) { \
      free(MAT_BUF(O1)); MAT_BUF(O1) = O2; MAT_ID(O1) = id; \
    } \
}

#define free_lists_exit(argI,argJ,I,J,ret) { \
    if (argI && !Matrix_Check(argI)) { Py_XDECREF(I); } \
    if (argJ && !Matrix_Check(argJ)) { Py_XDECREF(J); } \
    return ret; }


static int convert_mtx(matrix *src, void *dest, int id)
{
  if (PY_NUMBER((PyObject *)src))
    return convert_num[id](dest, src, 1, 0);

  if (MAT_ID(src) == id) {
    memcpy(dest, src->buffer, E_SIZE[src->id]*MAT_LGT(src) );
    return 0;
  }

  int_t i;
  for (i=0; i<MAT_LGT(src); i++)
    if (convert_num[id](dest + i*E_SIZE[id], src, 0,i)) return -1;

  return 0;
}

void * convert_mtx_alloc(matrix *src, int id)
{
  void *ptr;
  if (MAT_ID(src) == id) return MAT_BUF(src);

  if (!(ptr = malloc(E_SIZE[id]*MAT_LGT(src)))) return NULL;

  int_t i;
  for (i=0; i<MAT_LGT(src); i++)
    if (convert_num[id](ptr + i*E_SIZE[id], src, 0,i))
      { free(ptr); return NULL; }

  return ptr;
}


/*
  Creates an unpopulated "empty" matrix. In API
 */
matrix * Matrix_New(int_t nrows, int_t ncols, int id)
{
  matrix *a;
  if ((nrows < 0) || (ncols < 0) || (id < INT) || (id > COMPLEX)) {
    PyErr_BadInternalCall();
    return NULL;
  }

  if (!(a = (matrix *)matrix_tp.tp_alloc(&matrix_tp, 0)))
    return NULL;

  a->id = id; a->nrows = nrows; a->ncols = ncols;
  if (!(a->buffer =  calloc(nrows*ncols,E_SIZE[id]))) {
    a->ob_type->tp_free((PyObject*)a);
    return (matrix *)PyErr_NoMemory();
  }

  return a;
}

/*
  Creates a copy of matrix as a new object. In API.
 */
matrix *Matrix_NewFromMatrix(matrix *src, int id)
{
  matrix *a;

  if (PY_NUMBER((PyObject *)src))
    return Matrix_NewFromNumber(1, 1, id, src, 1);

  if (!(a = Matrix_New(src->nrows, src->ncols, id)))
    return (matrix *)PyErr_NoMemory();

  if (convert_mtx(src, a->buffer, id)) {
    Py_DECREF(a); PY_ERR_TYPE("illegal type conversion");
  }

  return a;
}

/*
  Creates a matrix from a PyArrayInterface.
 */
matrix *Matrix_NewFromArrayStruct(PyObject *obj, int id, int_t *ndim)
{
  PyObject *cobj = PyObject_GetAttrString(obj, "__array_struct__");
  PyArrayInterface *src =  (PyArrayInterface *)PyCObject_AsVoidPtr(cobj);

  if (src->version != 2)
    PY_ERR(PyExc_AssertionError, "unexpected format in array structure");

  if (src->nd != 1 && src->nd != 2)
    PY_ERR(PyExc_TypeError, "imported array must have 1 or 2 dimensions");

  int src_id;
  switch (src->typekind) {
  case 'i' : src_id = INT; break;
  case 'f' : src_id = DOUBLE; break;
  case 'c' : src_id = COMPLEX; break;
  default:
    Py_DECREF(cobj); PY_ERR_TYPE("invalid array type");
  }

  if (id == -1) id = src_id;
  if ((src_id > id) || (src->itemsize != E_SIZE[src_id])) {
    Py_DECREF(cobj);
    PY_ERR_TYPE("invalid array type");
  }

  /* XXX: revise flags check */
  if (!(src->flags & 0x001) && !(src->flags & 0x002)) {
    Py_DECREF(cobj);
    PY_ERR_TYPE("error converting array");
  }

  *ndim = src->nd;
  matrix *a = Matrix_New(src->shape[0], src->nd == 2 ? src->shape[1] : 1, id);
  if (!a) {
    Py_DECREF(cobj); return (matrix *)PyErr_NoMemory();
  }

  int_t i, j, cnt;

  for (j=0, cnt=0; j<MAT_NCOLS(a); j++) {
    for (i=0; i<src->shape[0]; i++, cnt++) {

      number n;
      switch (id) {
      case INT :
        MAT_BUFI(a)[cnt] =
            *(int_t *)(src->data+i*src->strides[0]+j*src->strides[1]);
        break;
      case DOUBLE:
        switch (src_id) {
        case INT:
          n.d = *(int_t *)(src->data + i*src->strides[0]+j*src->strides[1]);
          break;
        case DOUBLE:
          n.d = *(double *)(src->data + i*src->strides[0]+j*src->strides[1]);
          break;
        }
        MAT_BUFD(a)[cnt] = n.d;
        break;
        case COMPLEX:
          switch (src_id) {
          case INT:
            n.z = *(int_t *)(src->data+i*src->strides[0]+j*src->strides[1]);
            break;
          case DOUBLE:
            n.z = *(double *)(src->data+i*src->strides[0]+j*src->strides[1]);
            break;
          case COMPLEX:
            n.z = *(complex *)(src->data+i*src->strides[0]+j*src->strides[1]);
            break;
          }
          MAT_BUFZ(a)[cnt] = n.z;
          break;
      }
    }
  }

  Py_DECREF(cobj);
  return a;
}

/*
  Generates a matrix with all entries equal.
 */
matrix *
Matrix_NewFromNumber(int_t nrows, int_t ncols, int_t id, void *val, int val_id)
{

  int_t i;
  matrix *a = Matrix_New(nrows, ncols, id);
  if (!a) return (matrix *)PyErr_NoMemory();

  number n;
  if (convert_num[id](&n, val, val_id, 0)) { Py_DECREF(a); return NULL; }
  for (i=0; i<MAT_LGT(a); i++) write_num[id](MAT_BUF(a), i, &n, 0);

  return a;
}

/*
  Converts a Python list to a matrix. Part of API
 */
matrix * Matrix_NewFromSequence(PyObject *x, int id)
{
  int_t i, len = PySequence_Size(x);
  PyObject *seq = PySequence_Fast(x, "list is not iterable");
  if (!seq) return NULL;

  if (id == -1) {
    for (i=0; i<len; i++) {
      PyObject *item = PySequence_Fast_GET_ITEM(seq, i);

      if (!PY_NUMBER(item)) {
        Py_DECREF(seq); PY_ERR_TYPE("non-numeric element in list");
      }

      id = MAX(id, get_id(item, 1));
    }
  }

  if (!len) return Matrix_New(0, 1, (id < 0 ? INT : id));

  matrix *L = Matrix_New(len,1,id);
  if (!L) { Py_DECREF(seq); return (matrix *)PyErr_NoMemory(); }

  for (i=0; i<len; i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(seq, i);

    if (!PY_NUMBER(item)) {
      Py_DECREF(seq); Py_DECREF(L);
      PY_ERR_TYPE("non-numeric type in list");
    }

    number n;
    if (convert_num[id](&n, item, 1, 0)) {
      Py_DECREF(L); Py_DECREF(seq);
      PY_ERR(PyExc_TypeError, err_mtx_list2matrix[id]);
    }
    write_num[id](L->buffer, i, &n, 0);
  }
  Py_DECREF(seq);
  return L;
}

matrix * dense(spmatrix *self)
{
  matrix *A;
  int_t j, k;

  if (!(A = Matrix_New(SP_NROWS(self),SP_NCOLS(self),SP_ID(self))))
    return (matrix *)PyErr_NoMemory();

  if (SP_ID(self) == DOUBLE) {
    for (j=0; j<SP_NCOLS(self); j++)
      for (k=SP_COL(self)[j]; k<SP_COL(self)[j+1]; k++) {
        MAT_BUFD(A)[SP_ROW(self)[k] + j*MAT_NROWS(A)] = SP_VALD(self)[k];
      }
  } else {
    for (j=0; j<SP_NCOLS(self); j++)
      for (k=SP_COL(self)[j]; k<SP_COL(self)[j+1]; k++) {
        MAT_BUFZ(A)[SP_ROW(self)[k] + j*MAT_NROWS(A)] = SP_VALZ(self)[k];
      }
  }

  return A;
}

int convert_array(void *dest, void *src, int dest_id, int src_id, int n);

matrix * dense_concat(PyObject *L, int id_arg)
{
  int_t m=0, n=0, mk=0, nk=0, i=0, j, id = 0;
  PyObject *col;

  int single_col = (PyList_GET_SIZE(L) > 0 &&
      !PyList_Check(PyList_GET_ITEM(L, 0)));

  for (j=0; j<(single_col ? 1 : PyList_GET_SIZE(L)); j++) {

    col = (single_col ? L : PyList_GET_ITEM(L, j));
    if (!PyList_Check(col))
      PY_ERR_TYPE("invalid type in list");

    mk = 0;
    for (i=0; i<PyList_GET_SIZE(col); i++) {
      PyObject *Lij = PyList_GET_ITEM(col, i);
      if (!Matrix_Check(Lij) && !SpMatrix_Check(Lij) && !PY_NUMBER(Lij))
        PY_ERR_TYPE("invalid type in list");

      int blk_nrows, blk_ncols;
      if (Matrix_Check(Lij) || SpMatrix_Check(Lij)) {
        blk_nrows = X_NROWS(Lij); blk_ncols = X_NCOLS(Lij);
        id = MAX(id, X_ID(Lij));
      } else {
        blk_nrows = 1; blk_ncols = 1;
        id = MAX(id, get_id(Lij,1));
      }

      if (i==0) {
        nk = blk_ncols; n += nk;
        mk = blk_nrows;
      } else {
        if (blk_ncols != nk)
          PY_ERR_TYPE("incompatible dimensions of subblocks");
        mk += blk_nrows;
      }
    }
    if (j==0)
      m = mk;
    else if (m != mk) PY_ERR_TYPE("incompatible dimensions of subblocks");
  }

  if ((id_arg >= 0) && (id_arg < id))
    PY_ERR_TYPE("illegal type conversion");

  id = MAX(id, id_arg);

  matrix *A = Matrix_New(m, n, id);
  if (!A) return (matrix *)PyErr_NoMemory();

  nk = 0;
  for (j=0; j<(single_col ? 1 : PyList_GET_SIZE(L)); j++) {
    col = (single_col ? L : PyList_GET_ITEM(L, j));

    mk = 0;
    int blk_nrows = 0, blk_ncols = 0;
    for (i=0; i<PyList_GET_SIZE(col); i++) {
      PyObject *Lij = PyList_GET_ITEM(col, i);

      if (Matrix_Check(Lij) || SpMatrix_Check(Lij)) {
        blk_nrows = X_NROWS(Lij); blk_ncols = X_NCOLS(Lij);
      } else {
        blk_nrows = 1; blk_ncols = 1;
      }

      int ik, jk;
      for (jk=0; jk<blk_ncols; jk++) {

        if (Matrix_Check(Lij)) {
          for (ik=0; ik<blk_nrows; ik++)
            convert_num[id](MAT_BUF(A) + (mk+ik+(nk+jk)*m)*E_SIZE[id],
                Lij, 0, ik + jk*blk_nrows);

        } else if (SpMatrix_Check(Lij)) {
          for (ik=SP_COL(Lij)[jk]; ik<SP_COL(Lij)[jk+1]; ik++)
            convert_array(MAT_BUF(A) + ((nk+jk)*m+mk+SP_ROW(Lij)[ik])*
                E_SIZE[id], SP_VAL(Lij) + ik*E_SIZE[SP_ID(Lij)],
                id, SP_ID(Lij), 1);

        } else {

          convert_num[id](MAT_BUF(A) + (mk+(nk+jk)*m)*E_SIZE[id],
              Lij, 1, 0);
        }
      }
      mk += blk_nrows;
    }
    nk += blk_ncols;
  }
  return A;
}

static void
matrix_dealloc(matrix* self)
{
  free(self->buffer);
  self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
matrix_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  PyObject *Objx = NULL, *size = NULL;
  static char *kwlist[] = { "x", "size", "tc", NULL};

  int_t nrows=0, ncols=0;
  char tc = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOc:matrix", kwlist,
      &Objx, &size, &tc))
    return NULL;

  if (size && !PyArg_ParseTuple(size, "ll", &nrows, &ncols))
    PY_ERR_TYPE("invalid dimension tuple") ;

  if (nrows < 0 || ncols < 0)
    PY_ERR_TYPE("dimensions must be non-negative");

  if (tc && !(VALID_TC_MAT(tc))) PY_ERR_TYPE("tc must be 'i', 'd' or 'z'");
  int id = (tc ? TC2ID(tc) : -1);

  if (!Objx && size) PY_ERR_TYPE("invalid arguments");

  if (!Objx) return (PyObject *)Matrix_New(0, 0, (id == -1 ? INT : id));

  matrix *ret = NULL;
  /* x is a number */
  if (PY_NUMBER(Objx))
    return (PyObject *)
    Matrix_NewFromNumber(MAX(nrows, size ? 0 : 1),
        MAX(ncols, size ? 0 : 1), (id == -1 ? get_id(Objx,1):id),Objx,1);

  /* a matrix */
  else if (Matrix_Check(Objx))
    ret = Matrix_NewFromMatrix((matrix *)Objx, (id == -1 ?MAT_ID(Objx):id));

  /* sparse matrix */
  else if (SpMatrix_Check(Objx)) {
    matrix *tmp = dense((spmatrix *)Objx);
    if (!tmp) return PyErr_NoMemory();
    if (tmp->id != id) {
      ret = Matrix_NewFromMatrix(tmp, (id == -1 ? SP_ID(Objx) : id));
      Py_DECREF(tmp);
    } else {
      ret = tmp;
    }
  }

  /* PyArrayStructure */
  else if (PyObject_HasAttrString(Objx,"__array_struct__")) {
    int_t ndim = 0;
    ret = Matrix_NewFromArrayStruct(Objx, id, &ndim);
  }

  /* x is a list */
  else if (PyList_Check(Objx)) {

    /* first try a regular list */
    if (!(ret = Matrix_NewFromSequence(Objx, id))) {
      PyErr_Clear();
      /* try concatenation */
      ret = dense_concat(Objx, id);
    }
  }

  /* x is a sequence */
  else if (PySequence_Check(Objx)) {
    ret = Matrix_NewFromSequence(Objx, id);
  }
  else PY_ERR_TYPE("invalid matrix initialization");

  if (ret && size) {
    if (nrows*ncols == MAT_LGT(ret)) {
      ret->nrows=nrows; ret->ncols=ncols;
    } else {
      Py_DECREF(ret); PY_ERR_TYPE("wrong matrix dimensions");
    }
  }

  return (PyObject *)ret;
}

PyObject *matrix_sub(PyObject *, PyObject *);

static PyObject *
matrix_richcompare(PyObject *self, PyObject *other, int op) {

  PY_ERR(PyExc_NotImplementedError, "matrix comparison not implemented");
}

static PyObject *
matrix_str(matrix *self) {

  PyObject *cvxopt = PyImport_ImportModule("cvxopt");
  PyObject *str, *ret;

  if (!(str = PyObject_GetAttrString(cvxopt, "matrix_str"))) {
    Py_DECREF(cvxopt);
    PY_ERR(PyExc_KeyError, "missing 'matrix_str' in 'cvxopt'");
  }

  Py_DECREF(cvxopt);
  if (!PyCallable_Check(str)) PY_ERR_TYPE("'matrix_str' is not callable");

  ret = PyObject_CallFunctionObjArgs(str, (PyObject *)self, NULL);
  Py_DECREF(str);

  return ret;
}

static PyObject *
matrix_repr(matrix *self) {

  PyObject *cvxopt = PyImport_ImportModule("cvxopt");
  PyObject *repr, *ret;

  if (!(repr = PyObject_GetAttrString(cvxopt, "matrix_repr"))) {
    Py_DECREF(cvxopt);
    PY_ERR(PyExc_KeyError, "missing 'matrix_repr' in 'cvxopt'");
  }

  Py_DECREF(cvxopt);
  if (!PyCallable_Check(repr)) PY_ERR_TYPE("'matrix_repr' is not callable");

  ret = PyObject_CallFunctionObjArgs(repr, (PyObject *)self, NULL);
  Py_DECREF(repr);

  return ret;
}

/*
 * This method converts different index sets into a matrix indexlist
 */
matrix * create_indexlist(int_t dim, PyObject *A)
{
  matrix *x;
  int_t i, j;

  /* integer */
  if (PyInt_Check(A)) {
    i = PyInt_AS_LONG(A);
    if (OUT_RNG(i,dim)) PY_ERR(PyExc_IndexError, "index out of range");

    if ((x = Matrix_New(1,1,INT))) MAT_BUFI(x)[0] = i;
    return x;
  }
  /* slice */
  else if (PySlice_Check(A)) {
    int_t start, stop, step, lgt;

    if (PySlice_GetIndicesEx((PySliceObject*)A, dim,
        &start, &stop, &step, &lgt) < 0) return NULL;

    if ((x = Matrix_New(lgt, 1, INT)))
      for (i=start, j=0; j<lgt; i += step, j++) MAT_BUFI(x)[j] = i;
    else {
      return (matrix *)PyErr_NoMemory();
    }
    return x;
  }
  /* Matrix index list */
  else if (Matrix_Check(A)) {
    if (MAT_ID(A) != INT) PY_ERR_TYPE("not an integer index list");

    for (i=0; i<MAT_LGT(A); i++)
      if ( OUT_RNG(MAT_BUFI(A)[i], dim) )
        PY_ERR(PyExc_IndexError, "index out of range");

    return (matrix *)A;
  }
  /* List */
  else if (PyList_Check(A)) {
    if (!(x = (matrix *)Matrix_NewFromSequence(A, INT))) return NULL;

    return create_indexlist(dim, (PyObject *)x);
  }
  else PY_ERR(PyExc_TypeError, "invalid index argument");
}

static int
matrix_length(matrix *self)
{
  return MAT_LGT(self);
}

static PyObject *
matrix_subscr(matrix* self, PyObject* args)
{
  matrix *Il = NULL, *Jl = NULL, *ret;
  if (PyInt_Check(args)) {
    int_t i = PyInt_AS_LONG(args);
    if (OUT_RNG(i,MAT_LGT(self)))
      PY_ERR(PyExc_IndexError, "index out of range");

    return num2PyObject[self->id](self->buffer, CWRAP(i,MAT_LGT(self)));
  }

  else if (PyList_Check(args) || Matrix_Check(args) || PySlice_Check(args)) {

    if (!(Il = create_indexlist(MAT_LGT(self), args))) return NULL;

    int i;
    if (!(ret = Matrix_New(MAT_LGT(Il), 1, self->id) ))
      free_lists_exit(args,(PyObject *)NULL,Il,(PyObject *)NULL,
          PyErr_NoMemory());

    for (i=0; i<MAT_LGT(Il); i++)
      write_num[self->id](ret->buffer, i, self->buffer,
          CWRAP(MAT_BUFI(Il)[i],MAT_LGT(self)));

    free_lists_exit(args,(PyObject *)NULL,Il,(PyObject *)NULL,(PyObject *)ret);
  }

  /* remainding cases are different two argument indexing */
  PyObject *argI = NULL, *argJ = NULL;
  if (!PyArg_ParseTuple(args, "OO", &argI,&argJ))
    PY_ERR_TYPE("invalid index arguments");

  /* handle normal subscripts (two integers) separately */
  if (PyInt_Check(argI) && PyInt_Check(argJ)) {

    int i = PyInt_AS_LONG(argI), j = PyInt_AS_LONG(argJ);
    if ( OUT_RNG(i, self->nrows) || OUT_RNG(j, self->ncols))
      PY_ERR(PyExc_IndexError, "index out of range");

    return num2PyObject[self->id](self->buffer,
        CWRAP(i,self->nrows) + CWRAP(j,self->ncols)*self->nrows);
  }

  /* two slices, handled separately for speed */
  if (PySlice_Check(argI) && PySlice_Check(argJ)) {
    int_t rowstart, rowstop, rowstep, rowlgt;
    int_t colstart, colstop, colstep, collgt;

    if ( (PySlice_GetIndicesEx((PySliceObject*)argI, MAT_NROWS(self),
        &rowstart, &rowstop, &rowstep, &rowlgt) < 0) ||
        (PySlice_GetIndicesEx((PySliceObject*)argJ, MAT_NCOLS(self),
            &colstart, &colstop, &colstep, &collgt) < 0)) return NULL;

    if (!(ret = Matrix_New(rowlgt, collgt, self->id)))
      return PyErr_NoMemory();

    int i, j, icnt, jcnt, cnt=0;
    for (j=colstart, jcnt=0; jcnt<collgt; jcnt++, j+=colstep)
      for (i=rowstart, icnt=0; icnt<rowlgt; icnt++, i+=rowstep) {
        switch (self->id) {
        case INT:
          MAT_BUFI(ret)[cnt++] = MAT_BUFI(self)[j*self->nrows+i];
          break;
        case DOUBLE:
          MAT_BUFD(ret)[cnt++] = MAT_BUFD(self)[j*self->nrows+i];
          break;
        case COMPLEX:
          MAT_BUFZ(ret)[cnt++] = MAT_BUFZ(self)[j*self->nrows+i];
          break;
        }
      }

    return (PyObject *)ret;
  }

  /* remaining two indexing cases */
  if (!(Il = create_indexlist(self->nrows, argI)) ||
      !(Jl = create_indexlist(self->ncols, argJ)))
    free_lists_exit(argI, argJ, Il, Jl, (PyObject *)NULL);

  int i, j, cnt;
  if (!(ret = Matrix_New(MAT_LGT(Il), MAT_LGT(Jl), self->id)))
    free_lists_exit(argI, argJ, Il, Jl, PyErr_NoMemory());

  for (j=0, cnt=0; j < MAT_LGT(Jl); j++)
    for (i=0; i < MAT_LGT(Il); i++) {
      write_num[self->id](ret->buffer, cnt++, self->buffer,
          CWRAP(MAT_BUFI(Il)[i],self->nrows) +
          CWRAP(MAT_BUFI(Jl)[j],self->ncols)*self->nrows);
    }

  free_lists_exit(argI, argJ, Il, Jl, (PyObject *)ret);
}

#define spmatrix_getitem_i(O,i,v) \
    spmatrix_getitem_ij(O,i%SP_NROWS(O),i/SP_NROWS(O),v)

int spmatrix_getitem_ij(spmatrix *, int_t, int_t, number *) ;

static int
matrix_ass_subscr(matrix* self, PyObject* args, PyObject* val)
{
  matrix *Il = NULL, *Jl = NULL;
  int_t i, j, id = self->id, decref_val = 0, arraystruct_nd = 0;

  if (!val) PY_ERR_INT(PyExc_NotImplementedError,
      "cannot delete matrix entries");

  if (!(PY_NUMBER(val) || Matrix_Check(val) || SpMatrix_Check(val))) {

    if (PyObject_HasAttrString(val,"__array_struct__"))
      val = (PyObject *)Matrix_NewFromArrayStruct(val, -1, &arraystruct_nd);
    else
      val = (PyObject *)Matrix_NewFromSequence(val, MAT_ID(self));

    if (!val)
      PY_ERR_INT(PyExc_NotImplementedError, "invalid type in assignment");

    decref_val = 1;
  }

  if (get_id(val, (PY_NUMBER(val) ? 1 : 0)) > id)
    PY_ERR_INT(PyExc_TypeError, "invalid type in assignment");

  if (PyInt_Check(args) || PyList_Check(args) ||
      Matrix_Check(args) || PySlice_Check(args)) {

    if (!(Il = create_indexlist(MAT_LGT(self), args))) {
      if (decref_val) { Py_DECREF(val); }
      return -1;
    }
    number n;
    if (PY_NUMBER(val) || (Matrix_Check(val) && MAT_LGT(val)==1)) {
      convert_num[id](&n, val, (Matrix_Check(val) ? 0 : 1), 0);

      for (i=0; i<MAT_LGT(Il); i++)
        write_num[id](self->buffer,CWRAP(MAT_BUFI(Il)[i],MAT_LGT(self)),&n,0);
    }
    else {

      if (Matrix_Check(val)) {
        if (MAT_LGT(val) != MAT_LGT(Il) || MAT_NCOLS(val) > 1) {
          if (!Matrix_Check(args)) { Py_DECREF(Il); }
          if (decref_val) { Py_DECREF(val); }
          PY_ERR_INT(PyExc_TypeError, "argument has wrong size");
        }

        for (i=0; i < MAT_LGT(Il); i++) {
          convert_num[id](&n, val, 0, i);
          write_num[id](self->buffer,
              CWRAP(MAT_BUFI(Il)[i], MAT_LGT(self)), &n, 0);
        }
      }
      else { /* spmatrix */

        if (SP_NROWS(val) != MAT_LGT(Il) || SP_NCOLS(val) > 1) {
          if (!Matrix_Check(args)) { Py_DECREF(Il); }
          if (decref_val) { Py_DECREF(val); }
          PY_ERR_INT(PyExc_TypeError, "argument has wrong size");
        }

        for (i=0; i < MAT_LGT(Il); i++) {
          spmatrix_getitem_i((spmatrix *)val, i, &n);
          write_num[id](self->buffer,
              CWRAP(MAT_BUFI(Il)[i], MAT_LGT(self)), &n, 0);
        }
      }
    }

    if (decref_val) { Py_DECREF(val); }
    free_lists_exit(args,(PyObject *)NULL,Il,(PyObject *)NULL,0);
  }

  /* remainding cases are different two argument indexing */
  PyObject *argI = NULL, *argJ = NULL;
  if (!PyArg_ParseTuple(args, "OO", &argI,&argJ))
    PY_ERR_INT(PyExc_TypeError, "invalid index arguments");

  /* two slices, RHS of same type, handled separately for speed */
  if (PySlice_Check(argI) && PySlice_Check(argJ) &&
      Matrix_Check(val) && MAT_ID(val) == MAT_ID(self)) {

    int_t rowstart, rowstop, rowstep, rowlgt;
    int_t colstart, colstop, colstep, collgt;

    if ( (PySlice_GetIndicesEx((PySliceObject*)argI, MAT_NROWS(self),
        &rowstart, &rowstop, &rowstep, &rowlgt) < 0) ||
        (PySlice_GetIndicesEx((PySliceObject*)argJ, MAT_NCOLS(self),
            &colstart, &colstop, &colstep, &collgt) < 0)) return -1;

    if (decref_val && MAT_LGT(val) == rowlgt*collgt) {
      MAT_NROWS(val) = rowlgt; MAT_NCOLS(val) = collgt;
    }

    if (MAT_NROWS(val)!=rowlgt || MAT_NCOLS(val)!=collgt)
      PY_ERR_INT(PyExc_TypeError, "argument has wrong size");

    int i, j, icnt, jcnt, cnt=0;
    for (j=colstart, jcnt=0; jcnt<collgt; jcnt++, j+=colstep)
      for (i=rowstart, icnt=0; icnt<rowlgt; icnt++, i+=rowstep) {
        switch (self->id) {
        case INT:
          MAT_BUFI(self)[j*self->nrows+i] = MAT_BUFI(val)[cnt++];
          break;
        case DOUBLE:
          MAT_BUFD(self)[j*self->nrows+i] = MAT_BUFD(val)[cnt++];
          break;
        case COMPLEX:
          MAT_BUFZ(self)[j*self->nrows+i] = MAT_BUFZ(val)[cnt++];
          break;
        }
      }

    return 0;
  }

  if (!(Il = create_indexlist(self->nrows, argI)) ||
      !(Jl = create_indexlist(self->ncols, argJ))) {
    if (decref_val) { Py_DECREF(val); }
    free_lists_exit(argI,argJ,Il,Jl,-1);
  }

  if (decref_val && arraystruct_nd < 2 &&
      MAT_LGT(val) == MAT_LGT(Il)*MAT_LGT(Jl)) {
    MAT_NROWS(val) = MAT_LGT(Il); MAT_NCOLS(val) = MAT_LGT(Jl);
  }

  number n;
  if (PY_NUMBER(val) || (Matrix_Check(val) && MAT_LGT(val)==1)) {
    if (convert_num[id](&n, val, (Matrix_Check(val) ? 0 : 1), 0)) {
      if (decref_val) { Py_DECREF(val); }
      free_lists_exit(Il,Jl,argI,argJ,-1);
    }

    for (j=0; j < MAT_LGT(Jl); j++)
      for (i=0; i < MAT_LGT(Il); i++) {
        write_num[id](self->buffer,CWRAP(MAT_BUFI(Il)[i],self->nrows) +
            CWRAP(MAT_BUFI(Jl)[j],self->ncols)*self->nrows, &n, 0);
      }
  }
  else if (Matrix_Check(val)) {
    if (MAT_LGT(Il) != MAT_NROWS(val) || MAT_LGT(Jl) != MAT_NCOLS(val) ) {
      if (!Matrix_Check(argI)) { Py_DECREF(Il); }
      if (!Matrix_Check(argJ)) { Py_DECREF(Jl); }
      if (decref_val) { Py_DECREF(val); }
      PY_ERR_INT(PyExc_TypeError, "argument has wrong size");
    }

    int cnt = 0;
    for (j=0; j < MAT_LGT(Jl); j++)
      for (i=0; i < MAT_LGT(Il); i++, cnt++) {
        if (convert_num[id](&n, val, 0, cnt))
          free_lists_exit(argI,argJ,Il,Jl,-1);

        write_num[id](self->buffer,CWRAP(MAT_BUFI(Il)[i],self->nrows) +
            CWRAP(MAT_BUFI(Jl)[j],self->ncols)*self->nrows, &n, 0);
      }
  }
  else { /* spmatrix */
    if (MAT_LGT(Il) != SP_NROWS(val) || MAT_LGT(Jl) != SP_NCOLS(val) ) {
      if (!Matrix_Check(argI)) { Py_DECREF(Il); }
      if (!Matrix_Check(argJ)) { Py_DECREF(Jl); }
      if (decref_val) { Py_DECREF(val); }
      PY_ERR_INT(PyExc_TypeError, "argument has wrong size");
    }

    int cnt = 0;
    for (j=0; j < MAT_LGT(Jl); j++)
      for (i=0; i < MAT_LGT(Il); i++, cnt++) {

        spmatrix_getitem_i((spmatrix *)val, cnt, &n);
        write_num[id](self->buffer,CWRAP(MAT_BUFI(Il)[i],self->nrows)  +
            CWRAP(MAT_BUFI(Jl)[j],self->ncols)*self->nrows, &n, 0);
      }
  }

  if (decref_val) { Py_DECREF(val); }
  free_lists_exit(argI,argJ,Il,Jl,0);
}

static PyMappingMethods matrix_as_mapping = {
    (lenfunc)matrix_length,
    (binaryfunc)matrix_subscr,
    (objobjargproc)matrix_ass_subscr
};

static PyObject * matrix_transpose(matrix *self) {

  matrix *ret = Matrix_New(self->ncols, self->nrows, self->id);
  if (!ret) return PyErr_NoMemory();

  int i, j, cnt = 0;
  for (i=0; i < ret->nrows; i++)
    for (j=0; j < ret->ncols; j++)
      write_num[self->id](ret->buffer, i + j*ret->nrows, self->buffer, cnt++);

  return (PyObject *)ret;
}

static PyObject * matrix_ctranspose(matrix *self) {

  if (self->id != COMPLEX) return matrix_transpose(self);

  matrix *ret = Matrix_New(self->ncols, self->nrows, self->id);
  if (!ret) return PyErr_NoMemory();

  int i, j, cnt = 0;
  for (i=0; i < ret->nrows; i++)
    for (j=0; j < ret->ncols; j++)
      MAT_BUFZ(ret)[i + j*ret->nrows] = conj(MAT_BUFZ(self)[cnt++]);

  return (PyObject *)ret;
}

static PyObject * matrix_real(matrix *self) {

  if (self->id != COMPLEX)
    return (PyObject *)Matrix_NewFromMatrix(self,self->id);

  matrix *ret = Matrix_New(self->nrows, self->ncols, DOUBLE);
  if (!ret) return PyErr_NoMemory();

  int i;
  for (i=0; i < MAT_LGT(self); i++)
    MAT_BUFD(ret)[i] = creal(MAT_BUFZ(self)[i]);

  return (PyObject *)ret;
}

static PyObject * matrix_imag(matrix *self) {

  matrix *ret;
  if (self->id != COMPLEX) {
    PyObject *a = PyFloat_FromDouble(0);
    ret = Matrix_NewFromNumber(self->nrows, self->ncols, self->id, a, 2);
    Py_DECREF(a);
    if (!ret) return PyErr_NoMemory();

    return (PyObject *)ret;
  }

  if (!(ret = Matrix_New(self->nrows, self->ncols, DOUBLE)))
    return PyErr_NoMemory();

  int i;
  for (i=0; i < MAT_LGT(self); i++)
    MAT_BUFD(ret)[i] = cimag(MAT_BUFZ(self)[i]);

  return (PyObject *)ret;
}

static char doc_tofile[] =
    "Writes a matrix to file\n\n"
    "ARGUMENTS\n"
    "fo          a Python file object prevously obtained by open()\n\n";
static PyObject *
matrix_tofile(matrix *self, PyObject *args, PyObject *kwrds)
{
  PyObject *file_obj;
  FILE *fp;
  char *kwlist[] = {"fo",  NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O", kwlist, &file_obj))
    return NULL;

  if (!PyFile_Check(file_obj))
    PY_ERR_TYPE("argument must a file object");

  if (!(fp = PyFile_AsFile(file_obj)))
    PY_ERR(PyExc_IOError,"file not open for writing");

  if (fwrite(self->buffer, E_SIZE[self->id], MAT_LGT(self), fp)) ;
  return Py_BuildValue("");
}

static char doc_fromfile[] =
    "Reads a matrix from file\n\n"
    "ARGUMENTS\n"
    "fo          a Python file object prevously obtained by open()\n\n";
static PyObject *
matrix_fromfile(matrix *self, PyObject *args, PyObject *kwrds)
{
  PyObject *file_obj;
  FILE *fp;
  char *kwlist[] = {"fo", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O", kwlist, &file_obj))
    return NULL;

  if (!PyFile_Check(file_obj))
    PY_ERR_TYPE("argument must a file object");
  if (!(fp = PyFile_AsFile(file_obj)))
    PY_ERR(PyExc_IOError,"file not open for reading");

  int n = fread(self->buffer, E_SIZE[self->id], MAT_LGT(self), fp);

  if (n < MAT_LGT(self))
    PY_ERR(PyExc_IOError, "could not read entire matrix");
  return Py_BuildValue("");
}

static PyObject *
matrix_getstate(matrix *self)
{
  PyObject *list = PyList_New(MAT_LGT(self));
  PyObject *size = PyTuple_New(2);
  if (!list || !size) {
    Py_XDECREF(list); Py_XDECREF(size); return NULL;
  }

  PyTuple_SET_ITEM(size, 0, PyInt_FromLong(MAT_NROWS(self)));
  PyTuple_SET_ITEM(size, 1, PyInt_FromLong(MAT_NCOLS(self)));

  int i;
  for (i=0; i<MAT_LGT(self); i++) {
    PyList_SET_ITEM(list, i, num2PyObject[MAT_ID(self)](MAT_BUF(self), i));
  }

  return Py_BuildValue("NNs", list, size, TC_CHAR[MAT_ID(self)]);
}

static PyObject *
matrix_reduce(matrix* self)
{
  return Py_BuildValue("ON", self->ob_type, matrix_getstate(self));
}

static PyMethodDef matrix_methods[] = {
    {"trans", (PyCFunction)matrix_transpose, METH_NOARGS,
        "Returns the matrix transpose"},
        {"ctrans", (PyCFunction)matrix_ctranspose, METH_NOARGS,
            "Returns the matrix conjugate transpose"},
            {"real", (PyCFunction)matrix_real, METH_NOARGS,
                "Returns real part of matrix"},
                {"imag", (PyCFunction)matrix_imag, METH_NOARGS,
                    "Returns imaginary part of matrix"},
                    {"tofile", (PyCFunction)matrix_tofile, METH_VARARGS|METH_KEYWORDS, doc_tofile},
                    {"fromfile", (PyCFunction)matrix_fromfile, METH_VARARGS|METH_KEYWORDS, doc_fromfile},
                    {"__reduce__", (PyCFunction)matrix_reduce, METH_NOARGS, "__reduce__() -> (cls, state)"},
                    {NULL}  /* Sentinel */
};

static PyObject *
matrix_get_size(matrix *self, void *closure)
{
  PyObject *t = PyTuple_New(2);

  PyTuple_SET_ITEM(t, 0, PyInt_FromLong(self->nrows));
  PyTuple_SET_ITEM(t, 1, PyInt_FromLong(self->ncols));

  return t;
}

static int
matrix_set_size(matrix *self, PyObject *value, void *closure)
{
  if (value == NULL)
    PY_ERR_INT(PyExc_TypeError, "size attribute cannot be deleted");

  if (!PyTuple_Check(value) || PyTuple_Size(value) != 2)
    PY_ERR_INT(PyExc_TypeError, "can only assign a 2-tuple to size");

  if (!PyInt_Check(PyTuple_GET_ITEM(value, 0)) ||
      !PyInt_Check(PyTuple_GET_ITEM(value, 1)))
    PY_ERR_INT(PyExc_TypeError, "invalid size tuple");

  int m = PyInt_AS_LONG(PyTuple_GET_ITEM(value, 0));
  int n = PyInt_AS_LONG(PyTuple_GET_ITEM(value, 1));

  if (m<0 || n<0)
    PY_ERR_INT(PyExc_TypeError, "dimensions must be non-negative");

  if (m*n != MAT_LGT(self))
    PY_ERR_INT(PyExc_TypeError, "number of elements in matrix cannot change");

  MAT_NROWS(self) = m;
  MAT_NCOLS(self) = n;

  return 0;
}

static PyObject * matrix_get_typecode(matrix *self, void *closure)
{
  return PyString_FromStringAndSize(TC_CHAR[self->id], 1);
}


static void matrix_free_array_struct(void *a_struct, void *descr)
{
  free(((PyArrayInterface *)a_struct)->shape);
  free(((PyArrayInterface *)a_struct)->strides);
  free(a_struct);
}

static PyObject * matrix_array_struct(matrix *self, void *closure) {

  PyArrayInterface *a = malloc(sizeof(PyArrayInterface));
  if (!a) return PyErr_NoMemory();

  a->shape = malloc(2*sizeof(int_t));
  a->strides = malloc(2*sizeof(int_t));
  if (!a->shape || !a->strides) {
    free(a->shape); free(a->strides); free(a);
    return PyErr_NoMemory();
  }

  a->version = 2;
  a->nd = 2;
  a->typekind = PY_ARRAY_TC[self->id];
  a->itemsize = E_SIZE[self->id];
  a->flags = 0x001 + 0x002 + 0x100 + 0x200 + 0x400;
  a->shape[0] = self->nrows;
  a->shape[1] = self->ncols;
  a->strides[0] = E_SIZE[self->id];
  a->strides[1] = self->nrows*E_SIZE[self->id];
  a->data = self->buffer;

  return (PyObject *) PyCObject_FromVoidPtrAndDesc((void *) a,
      "CVXOPT ARRAY STRUCT", matrix_free_array_struct);

}

static PyObject * matrix_get_T(matrix *self, void *closure)
{
  return matrix_transpose(self);
}

static PyObject * matrix_get_H(matrix *self, void *closure)
{
  return matrix_ctranspose(self);
}

static PyGetSetDef matrix_getsets[] = {
    {"size", (getter) matrix_get_size, (setter) matrix_set_size,
        "matrix dimensions"},
        {"typecode", (getter) matrix_get_typecode, NULL, "typecode character"},
        {"__array_struct__", (getter) matrix_array_struct, NULL,
            "C object implementing the NumPy array protocol"},
            {"T", (getter) matrix_get_T, NULL, "transpose"},
            {"H", (getter) matrix_get_H, NULL, "conjugate transpose"},
            {NULL}  /* Sentinel */
};

PyTypeObject matrix_tp = {
    PyObject_HEAD_INIT(NULL)
    0,
    "cvxopt.base.matrix",
    sizeof(matrix),
    0,
    (destructor)matrix_dealloc,	        /* tp_dealloc */
    0,	               	                /* tp_print */
    0,					/* tp_getattr */
    0,					/* tp_setattr */
    0,					/* tp_compare */
    (reprfunc)matrix_repr,	                /* tp_repr */
    &matrix_as_number,			/* tp_as_number */
    0,	                                /* tp_as_sequence */
    &matrix_as_mapping,   	                /* tp_as_mapping */
    0,					/* tp_hash */
    0,					/* tp_call */
    (reprfunc)matrix_str,			/* tp_str */
    0,		                        /* tp_getattro */
    0,			                /* tp_setattro */
    0,			                /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
    Py_TPFLAGS_CHECKTYPES,                  /* tp_flags */
    0,					/* tp_doc */
    0,					/* tp_traverse */
    0,					/* tp_clear */
    (richcmpfunc)matrix_richcompare,        /* tp_richcompare */
    0,					/* tp_weaklistoffset */
    (getiterfunc)matrix_iter,		/* tp_iter */
    0,	       	                        /* tp_iternext */
    matrix_methods,		                /* tp_methods */
    0,			        	/* tp_members */
    matrix_getsets,			        /* tp_getset */
    0,		                        /* tp_base */
    0,					/* tp_dict */
    0,					/* tp_descr_get */
    0,					/* tp_descr_set */
    0,					/* tp_dictoffset */
    0,                                      /* tp_init */
    0,              			/* tp_alloc */
    matrix_new,				/* tp_new */
    0,           				/* tp_free */
};

/**************************************************************************/

static PyObject *
matrix_add_generic(PyObject *self, PyObject *other, int inplace)
{
  if (!(Matrix_Check(self) || PY_NUMBER(self)) ||
      !(Matrix_Check(other) || PY_NUMBER(other))) {
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
  }

  int id_self = get_id(self, (Matrix_Check(self) ? 0 : 1));
  int id_other = get_id(other, (Matrix_Check(other) ? 0 : 1));
  int id = MAX(id_self,id_other);

  if (inplace && (id != id_self || (MAT_LGT(self)==1 &&
      (Matrix_Check(other) && MAT_LGT(other)!=1))))
    PY_ERR_TYPE("invalid inplace operation");

  /* first operand is a scalar */
  if (PY_NUMBER(self) || (Matrix_Check(self) && MAT_LGT(self)==1))
    {
    number n;
    if (!inplace) {
      convert_num[id](&n,self,(Matrix_Check(self) ? 0 : 1),0);

      matrix *ret = Matrix_NewFromMatrix((matrix *)other, id);
      if (!ret) return PyErr_NoMemory();

      int lgt = MAT_LGT(ret), int1 = 1, int0 = 0;
      axpy[id](&lgt, &One[id], &n, &int0, ret->buffer, &int1);
      return (PyObject *)ret;
    }
    else {
      convert_num[id](&n,other,(Matrix_Check(other) ? 0 : 1),0);

      int int1 = 1, int0 = 0;
      axpy[id](&int1, &One[id], &n, &int0, MAT_BUF(self), &int1);

      Py_INCREF(self);
      return self;
    }
    }
  /* second operand is a scalar */
  else if (PY_NUMBER(other) || (Matrix_Check(other) &&
      MAT_LGT(other)==1))
    {
    number n;
    convert_num[id](&n,other,(Matrix_Check(other) ? 0 : 1),0);

    if (!inplace) {
      matrix *ret = Matrix_NewFromMatrix((matrix *)self, id);
      if (!ret) return PyErr_NoMemory();

      int lgt = MAT_LGT(self), int1 = 1, int0 = 0;
      axpy[id](&lgt, &One[id], &n, &int0, ret->buffer, &int1);

      return (PyObject *)ret;
    }
    else {
      int lgt = MAT_LGT(self), int1 = 1, int0 = 0;
      axpy[id](&lgt, &One[id], &n, &int0, MAT_BUF(self), &int1);

      Py_INCREF(self);
      return self;
    }
    }
  else { /* adding two matrices */
    if (MAT_NROWS(self) != MAT_NROWS(other) ||
        MAT_NCOLS(self) != MAT_NCOLS(other))
      PY_ERR_TYPE("incompatible dimensions");

    void *other_coerce = convert_mtx_alloc((matrix *)other, id);
    if (!other_coerce) return PyErr_NoMemory();

    if (!inplace) {
      matrix *ret = Matrix_NewFromMatrix((matrix *)self, id);
      if (!ret) return PyErr_NoMemory();

      int lgt = MAT_LGT(self), int1 = 1;
      axpy[id](&lgt, &One[id], other_coerce, &int1, MAT_BUF(ret), &int1);

      if (MAT_BUF(other) != other_coerce) { free(other_coerce); }
      return (PyObject *)ret;
    }
    else {
      int lgt = MAT_LGT(self), int1 = 1;
      axpy[id](&lgt, &One[id], other_coerce, &int1, MAT_BUF(self), &int1);

      if (MAT_BUF(other) != other_coerce) { free(other_coerce); }

      Py_INCREF(self);
      return self;
    }
  }
}

PyObject * matrix_add(PyObject *self, PyObject *other)
{
  return matrix_add_generic(self, other, 0);
}

static PyObject * matrix_iadd(PyObject *self,PyObject *other)
{
  return matrix_add_generic(self, other, 1);
}

static PyObject *
matrix_sub_generic(PyObject *self, PyObject *other, int inplace)
{
  if (!(Matrix_Check(self) || PY_NUMBER(self)) ||
      !(Matrix_Check(other) || PY_NUMBER(other))) {
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
  }

  int id_self = get_id(self, (Matrix_Check(self) ? 0 : 1));
  int id_other = get_id(other, (Matrix_Check(other) ? 0 : 1));
  int id = MAX(id_self,id_other);

  if (inplace && (id != id_self || (MAT_LGT(self)==1 &&
      (Matrix_Check(other) && MAT_LGT(other)!=1))))
    PY_ERR_TYPE("invalid inplace operation");

  /* first operand is a scalar */
  if (PY_NUMBER(self) || (Matrix_Check(self) && MAT_LGT(self)==1))
    {

    number n;
    if (!inplace) {
      convert_num[id](&n,self,(Matrix_Check(self) ? 0 : 1),0);

      matrix *ret = Matrix_NewFromMatrix((matrix *)other, id);
      if (!ret) return PyErr_NoMemory();

      int lgt = MAT_LGT(ret), int1 = 1, int0 = 0;
      scal[id](&lgt, &MinusOne[id], ret->buffer, &int1);
      axpy[id](&lgt, &One[id], &n, &int0, ret->buffer, &int1);
      return (PyObject *)ret;
    }
    else {
      convert_num[id](&n,other,(Matrix_Check(other) ? 0 : 1),0);

      int int1 = 1, int0 = 0;
      axpy[id](&int1, &MinusOne[id], &n, &int0, MAT_BUF(self), &int1);

      Py_INCREF(self);
      return self;
    }
    }
  /* second operand is a scalar */
  else if (PY_NUMBER(other) || (Matrix_Check(other) &&  MAT_LGT(other)==1))
    {
    number n;
    convert_num[id](&n,other,(Matrix_Check(other) ? 0 : 1),0);

    if (!inplace) {
      matrix *ret = Matrix_NewFromMatrix((matrix *)self, id);
      if (!ret) return PyErr_NoMemory();

      int lgt = MAT_LGT(self), int1 = 1, int0 = 0;
      axpy[id](&lgt, &MinusOne[id], &n, &int0, ret->buffer, &int1);

      return (PyObject *)ret;
    }
    else {
      int lgt = MAT_LGT(self), int1 = 1, int0 = 0;
      axpy[id](&lgt, &MinusOne[id], &n, &int0, MAT_BUF(self), &int1);

      Py_INCREF(self);
      return self;
    }
    }
  else { /* subtracting two matrices */
    if (MAT_NROWS(self) != MAT_NROWS(other) ||
        MAT_NCOLS(self) != MAT_NCOLS(other))
      PY_ERR_TYPE("incompatible dimensions");

    void *other_coerce = convert_mtx_alloc((matrix *)other, id);
    if (!other_coerce) return PyErr_NoMemory();

    if (!inplace) {
      matrix *ret = Matrix_NewFromMatrix((matrix *)self, id);
      if (!ret) return PyErr_NoMemory();

      int lgt = MAT_LGT(self), int1 = 1;
      axpy[id](&lgt, &MinusOne[id], other_coerce, &int1, MAT_BUF(ret), &int1);

      if (MAT_BUF(other) != other_coerce) { free(other_coerce); }
      return (PyObject *)ret;
    }
    else {
      int lgt = MAT_LGT(self), int1 = 1;
      axpy[id](&lgt,&MinusOne[id],other_coerce,&int1,MAT_BUF(self),&int1);

      if (MAT_BUF(other) != other_coerce) { free(other_coerce); }

      Py_INCREF(self);
      return self;
    }
  }
}

PyObject * matrix_sub(PyObject *self, PyObject *other)
{
  return matrix_sub_generic(self, other, 0);
}

static PyObject * matrix_isub(PyObject *self,PyObject *other)
{
  return matrix_sub_generic(self, other, 1);
}

static PyObject *
matrix_mul_generic(PyObject *self, PyObject *other, int inplace)
{
  if (!(Matrix_Check(self) || PY_NUMBER(self)) ||
      !(Matrix_Check(other) || PY_NUMBER(other))) {
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
  }

  int id_self = get_id(self, (Matrix_Check(self) ? 0 : 1));
  int id_other = get_id(other, (Matrix_Check(other) ? 0 : 1));
  int id = MAX(id_self,id_other);

  if (inplace && (id != id_self || (MAT_LGT(self)==1 &&
      (Matrix_Check(other) && MAT_LGT(other)!=1)) ||
      (MAT_LGT(self)>1 && (Matrix_Check(other) && MAT_LGT(other)>1))) )
    PY_ERR_TYPE("invalid inplace operation");

  /* first operand is a scalar */
  if (PY_NUMBER(self) || (Matrix_Check(self) && MAT_LGT(self)==1))
    {
    number n;
    if (!inplace) {
      convert_num[id](&n,self,(Matrix_Check(self) ? 0 : 1),0);

      matrix *ret = Matrix_NewFromMatrix((matrix *)other, id);
      if (!ret) return PyErr_NoMemory();

      int lgt = MAT_LGT(ret), int1 = 1;
      scal[id](&lgt, &n, ret->buffer, &int1);
      return (PyObject *)ret;
    }
    else {
      convert_num[id](&n,other,(Matrix_Check(other) ? 0 : 1),0);

      int int1 = 1;
      scal[id](&int1, &n, MAT_BUF(self), &int1);

      Py_INCREF(self);
      return self;
    }
    }
  /* second operand is a scalar */
  else if (PY_NUMBER(other) || (Matrix_Check(other) &&
      MAT_LGT(other)==1))
    {
    number n;
    convert_num[id](&n,other,(Matrix_Check(other) ? 0 : 1),0);

    if (!inplace) {
      matrix *ret = Matrix_NewFromMatrix((matrix *)self, id);
      if (!ret) return PyErr_NoMemory();

      int lgt = MAT_LGT(self), int1 = 1;
      scal[id](&lgt, &n, ret->buffer, &int1);
      return (PyObject *)ret;
    }
    else {
      int lgt = MAT_LGT(self), int1 = 1;
      scal[id](&lgt, &n, MAT_BUF(self), &int1);

      Py_INCREF(self);
      return self;
    }
    }
  else { /* multiplying two matrices */
    if (MAT_NCOLS(self) != MAT_NROWS(other))
      PY_ERR_TYPE("incompatible dimensions");

    int m = MAT_NROWS(self), n = MAT_NCOLS(other), k = MAT_NCOLS(self);

    int ldA = MAX(1,MAT_NROWS(self));
    int ldB = MAX(1,MAT_NROWS(other));
    int ldC = MAX(1,MAT_NROWS(self));
    char transA='N', transB='N';

    void *self_coerce = convert_mtx_alloc((matrix *)self, id);
    if (!self_coerce) return PyErr_NoMemory();

    void *other_coerce = convert_mtx_alloc((matrix *)other, id);
    if (!other_coerce) {
      if (MAT_ID(self) != id) { free(self_coerce); }
      return PyErr_NoMemory();
    }

    matrix *c = Matrix_New(m, n, id);
    if (!c) {
      if (MAT_ID(self) != id)  { free(self_coerce); }
      if (MAT_ID(other) != id) { free(other_coerce); }
      return PyErr_NoMemory();
    }

    gemm[id](&transA, &transB, &m, &n, &k, &One[id], self_coerce,
        &ldA, other_coerce, &ldB, &Zero[id], MAT_BUF(c), &ldC);

    if (MAT_ID(self) != id)  { free(self_coerce); }
    if (MAT_ID(other) != id) { free(other_coerce); }
    return (PyObject *)c;
  }
}

static PyObject * matrix_mul(PyObject *self, PyObject *other)
{
  return matrix_mul_generic(self, other, 0);
}

static PyObject * matrix_imul(PyObject *self,PyObject *other)
{
  return matrix_mul_generic(self, other, 1);
}

static PyObject *
matrix_div_generic(PyObject *self, PyObject *other, int inplace)
{
  if (!((Matrix_Check(other) && MAT_LGT(other)==1) || PY_NUMBER(other))) {
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
  }

  int id_self = get_id(self, (Matrix_Check(self) ? 0 : 1));
  int id_other = get_id(other, (Matrix_Check(other) ? 0 : 1));
  int id = MAX(id_self,id_other);

  number n;
  convert_num[id](&n,other,(Matrix_Check(other) ? 0 : 1),0);

  if (!inplace) {
    matrix *ret = Matrix_NewFromMatrix((matrix *)self, id);
    if (!ret) return PyErr_NoMemory();

    int lgt = MAT_LGT(ret);
    if (div_array[id](ret->buffer, n, lgt)) { Py_DECREF(ret); return NULL; }
    return (PyObject *)ret;
  }
  else {
    if (id != id_self) PY_ERR_TYPE("invalid inplace operation");

    if (div_array[id](MAT_BUF(self), n, MAT_LGT(self)))
      return NULL;

    Py_INCREF(self);
    return self;
  }
}

static PyObject * matrix_div(PyObject *self, PyObject *other)
{
  return matrix_div_generic(self, other, 0);
}

static PyObject * matrix_idiv(PyObject *self,PyObject *other)
{
  return matrix_div_generic(self, other, 1);
}

static PyObject *
matrix_rem_generic(PyObject *self, PyObject *other, int inplace)
{
  if (!((Matrix_Check(other) && MAT_LGT(other)==1) || PY_NUMBER(other))) {
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
  }

  int id_self = get_id(self, (Matrix_Check(self) ? 0 : 1));
  int id_other = get_id(other, (Matrix_Check(other) ? 0 : 1));
  int id = MAX(id_self,id_other);

  if (id == COMPLEX) PY_ERR(PyExc_NotImplementedError, "complex modulo");

  number n;
  convert_num[id](&n,other,(Matrix_Check(other) ? 0 : 1),0);

  if (!inplace) {
    matrix *ret = Matrix_NewFromMatrix((matrix *)self, id);
    if (!ret) return PyErr_NoMemory();

    int lgt = MAT_LGT(ret);
    if (mtx_rem[id](ret->buffer, n, lgt)) { Py_DECREF(ret); return NULL; }
    return (PyObject *)ret;
  }
  else {
    void *ptr = convert_mtx_alloc((matrix *)self, id);
    if (!ptr) return PyErr_NoMemory();

    int lgt = MAT_LGT(self);
    if (mtx_rem[id](ptr,n,lgt)) { free(ptr); return NULL; }

    free_convert_mtx_alloc(self, ptr, id);
    Py_INCREF(self);
    return self;
  }
}

static PyObject * matrix_rem(PyObject *self, PyObject *other)
{
  return matrix_rem_generic(self, other, 0);
}

static PyObject * matrix_irem(PyObject *self, PyObject *other)
{
  return matrix_rem_generic(self, other, 1);
}

static PyObject * matrix_neg(matrix *self)
{
  matrix *x = Matrix_NewFromMatrix(self,self->id);
  if (!x) return PyErr_NoMemory();

  int n = MAT_LGT(x), int1 = 1;
  scal[x->id](&n, &MinusOne[x->id], x->buffer, &int1);

  return (PyObject *)x;
}

static PyObject * matrix_pos(matrix *self)
{
  matrix *x = Matrix_NewFromMatrix(self, self->id);
  if (!x) return PyErr_NoMemory();

  return (PyObject *)x;
}

static PyObject * matrix_abs(matrix *self)
{
  matrix *ret = Matrix_New(self->nrows, self->ncols,
      (self->id == COMPLEX ? DOUBLE : self->id));

  if (!ret) return PyErr_NoMemory();

  mtx_abs[self->id](MAT_BUF(self), MAT_BUF(ret), MAT_LGT(self));
  return (PyObject *)ret;
}

static PyObject * matrix_pow(PyObject *self, PyObject *other)
{
  if (!PY_NUMBER(other)) PY_ERR_TYPE("exponent must be a number");

  number val;
  int id = MAX(DOUBLE, MAX(MAT_ID(self), get_id(other, 1)));
  convert_num[id](&val, other, 1, 0);
  matrix *Y = Matrix_NewFromMatrix((matrix *)self, id);
  if (!Y) return PyErr_NoMemory();

  int i;
  for (i=0; i<MAT_LGT(Y); i++) {
    if (id == DOUBLE) {
      if ((MAT_BUFD(Y)[i] == 0.0 && val.d < 0.0) ||
          (MAT_BUFD(Y)[i] < 0.0 && val.d < 1.0 && val.d > 0.0)) {
        Py_DECREF(Y);
        PY_ERR(PyExc_ValueError, "domain error");
      }


      MAT_BUFD(Y)[i] = pow(MAT_BUFD(Y)[i], val.d);
    } else {
      if (MAT_BUFZ(Y)[i] == 0.0 && (cimag(val.z) != 0.0 || creal(val.z)<0.0)) {
        Py_DECREF(Y);
        PY_ERR(PyExc_ValueError, "domain error");
      }
      MAT_BUFZ(Y)[i] = cpow(MAT_BUFZ(Y)[i], val.z);
    }
  }

  return (PyObject *)Y;
}

static int matrix_nonzero(matrix *self)
{
  int i, res = 0;
  for (i=0; i<MAT_LGT(self); i++) {
    if ((MAT_ID(self) == INT) && (MAT_BUFI(self)[i] != 0)) res = 1;
    else if ((MAT_ID(self) == DOUBLE) && (MAT_BUFD(self)[i] != 0)) res = 1;
    else if ((MAT_ID(self) == COMPLEX) && (MAT_BUFZ(self)[i] != 0)) res = 1;
  }

  return res;
}

static PyNumberMethods matrix_as_number = {
    (binaryfunc)matrix_add,	/*nb_add*/
    (binaryfunc)matrix_sub,	/*nb_subtract*/
    (binaryfunc)matrix_mul,	/*nb_multiply*/
    (binaryfunc)matrix_div, /*nb_divide*/
    (binaryfunc)matrix_rem, /*nb_remainder*/
    0,	                /*nb_divmod*/
    (ternaryfunc)matrix_pow, /*nb_power*/
    (unaryfunc)matrix_neg,	/*nb_negative*/
    (unaryfunc)matrix_pos,	/*nb_positive*/
    (unaryfunc)matrix_abs,	/*nb_absolute*/
    (inquiry)matrix_nonzero,/*nb_nonzero*/
    0,	                /*nb_invert*/
    0,	                /*nb_lshift*/
    0,	                /*nb_rshift*/
    0,	                /*nb_and*/
    0,	                /*nb_xor*/
    0,	                /*nb_or*/
    0,		        /*nb_coerce*/
    0,	                /*nb_int*/
    0,	                /*nb_long*/
    0,                      /*nb_float*/
    0,	                /*nb_oct*/
    0, 	                /*nb_hex*/
    (binaryfunc)matrix_iadd,/*nb_inplace_add*/
    (binaryfunc)matrix_isub,/*nb_inplace_subtract*/
    (binaryfunc)matrix_imul,/*nb_inplace_multiply*/
    (binaryfunc)matrix_idiv,/*nb_inplace_divide*/
    (binaryfunc)matrix_irem,/*nb_inplace_remainder*/
    0,			/*nb_inplace_power*/
    0,			/*nb_inplace_lshift*/
    0,			/*nb_inplace_rshift*/
    0,			/*nb_inplace_and*/
    0,			/*nb_inplace_xor*/
    0,			/*nb_inplace_or*/
    0,	                /*nb_floor_divide*/
    0,	                /*nb_true_divide*/
    0,			/*nb_inplace_floor_divide*/
    0,			/*nb_inplace_true_divide*/
};


/*********************** Iterator **************************/

typedef struct {
  PyObject_HEAD
  long index;
  matrix *mObj;
} matrixiter;

static PyTypeObject matrixiter_tp;

#define MatrixIter_Check(O) PyObject_TypeCheck(O, &matrixiter_tp)

static PyObject *
matrix_iter(matrix *obj)
{
  matrixiter *it;

  if (!Matrix_Check(obj)) {
    PyErr_BadInternalCall();
    return NULL;
  }

  it = PyObject_GC_New(matrixiter, &matrixiter_tp);
  if (!it) return NULL;

  matrixiter_tp.tp_iter = PyObject_SelfIter;
  matrixiter_tp.tp_getattro = PyObject_GenericGetAttr;

  Py_INCREF(obj);
  it->index = 0;
  it->mObj = obj;
  PyObject_GC_Track(it);

  return (PyObject *)it;
}

static void
matrixiter_dealloc(matrixiter *it)
{
  PyObject_GC_UnTrack(it);
  Py_XDECREF(it->mObj);
  PyObject_GC_Del(it);
}

static int
matrixiter_traverse(matrixiter *it, visitproc visit, void *arg)
{
  if (it->mObj == NULL)
    return 0;

  return visit((PyObject *)(it->mObj), arg);
}

static PyObject *
matrixiter_next(matrixiter *it)
{
  assert(MatrixIter_Check(it));
  if (it->index >= MAT_LGT(it->mObj))
    return NULL;

  return num2PyObject[it->mObj->id](it->mObj->buffer, it->index++);
}

static PyTypeObject matrixiter_tp = {
    PyObject_HEAD_INIT(NULL)
    0,					/* ob_size */
    "matrixiter",			        /* tp_name */
    sizeof(matrixiter),      		/* tp_basicsize */
    0,					/* tp_itemsize */
    (destructor)matrixiter_dealloc,		/* tp_dealloc */
    0,					/* tp_print */
    0,					/* tp_getattr */
    0,					/* tp_setattr */
    0,					/* tp_compare */
    0,					/* tp_repr */
    0,					/* tp_as_number */
    0,					/* tp_as_sequence */
    0,					/* tp_as_mapping */
    0,					/* tp_hash */
    0,					/* tp_call */
    0,					/* tp_str */
    0,                      		/* tp_getattro */
    0,					/* tp_setattro */
    0,					/* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,/* tp_flags */
    0,					/* tp_doc */
    (traverseproc)matrixiter_traverse,	/* tp_traverse */
    0,					/* tp_clear */
    0,					/* tp_richcompare */
    0,					/* tp_weaklistoffset */
    0,                  			/* tp_iter */
    (iternextfunc)matrixiter_next,		/* tp_iternext */
    0,					/* tp_methods */
};



PyObject * matrix_log(matrix *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A;

  if (!PyArg_ParseTuple(args, "O", &A)) return NULL;

  if (PyInt_Check(A) || PyFloat_Check(A)) {
    double f = PyFloat_AsDouble(A);
    if (f>0.0)
      return Py_BuildValue("d",log(f));
    else
      PY_ERR(PyExc_ValueError, "domain error");
  }
  else if (PyComplex_Check(A)) {
    number n;
    convert_num[COMPLEX](&n, A, 1, 0);

    if (n.z == 0) PY_ERR(PyExc_ValueError, "domain error");

    n.z = clog(n.z);
    return num2PyObject[COMPLEX](&n, 0);
  }

  else if (Matrix_Check(A)  && (MAT_ID(A) == INT || MAT_ID(A) == DOUBLE)) {
    if (MAT_LGT(A) == 0)
      return (PyObject *)Matrix_New(MAT_NROWS(A),MAT_NCOLS(A),DOUBLE);

    double val = (MAT_ID(A) == INT ? MAT_BUFI(A)[0] : MAT_BUFD(A)[0]);

    int i;
    for (i=1; i<MAT_LGT(A); i++) {
      if (MAT_ID(A) == INT)
        val = MIN(val,(MAT_BUFI(A)[i]));
      else
        val = MIN(val,(MAT_BUFD(A)[i]));
    }

    if (val > 0.0) {
      matrix *ret = Matrix_New(MAT_NROWS(A), MAT_NCOLS(A), DOUBLE);
      if (!ret) return PyErr_NoMemory();

      for (i=0; i<MAT_LGT(A); i++)
        MAT_BUFD(ret)[i] = log((MAT_ID(A)== INT ?
            MAT_BUFI(A)[i] : MAT_BUFD(A)[i]));

      return (PyObject *)ret;
    }
    else PY_ERR(PyExc_ValueError, "domain error");

  }
  else if (Matrix_Check(A) && MAT_ID(A) == COMPLEX) {
    matrix *ret = Matrix_New(MAT_NROWS(A), MAT_NCOLS(A), COMPLEX);
    if (!ret) return PyErr_NoMemory();

    int i;
    for (i=0; i<MAT_LGT(A); i++) {
      if (MAT_BUFZ(A)[i] == 0) {
        Py_DECREF(ret);
        PY_ERR(PyExc_ValueError, "domain error");
      }
      MAT_BUFZ(ret)[i] = clog(MAT_BUFZ(A)[i]);
    }
    return (PyObject *)ret;
  }
  else PY_ERR_TYPE("argument must a be a number or dense matrix");
}

PyObject * matrix_exp(matrix *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A;

  if (!PyArg_ParseTuple(args, "O", &A)) return NULL;

  if (PyInt_Check(A) || PyFloat_Check(A))
    return Py_BuildValue("d",exp(PyFloat_AsDouble(A)));

  else if (PyComplex_Check(A)) {
    number n;
    convert_num[COMPLEX](&n, A, 1, 0);
    n.z = cexp(n.z);
    return num2PyObject[COMPLEX](&n, 0);
  }

  else if (Matrix_Check(A)) {
    matrix *ret = Matrix_New(MAT_NROWS(A),MAT_NCOLS(A),
        (MAT_ID(A) == COMPLEX ? COMPLEX : DOUBLE));
    if (!ret) return PyErr_NoMemory();

    int i;
    if (MAT_ID(ret) == DOUBLE)
      for (i=0; i<MAT_LGT(ret); i++)
        MAT_BUFD(ret)[i] = exp(MAT_ID(A) == DOUBLE ? MAT_BUFD(A)[i] :
        MAT_BUFI(A)[i]);
    else
      for (i=0; i<MAT_LGT(ret); i++)
        MAT_BUFZ(ret)[i] = cexp(MAT_BUFZ(A)[i]);

    return (PyObject *)ret;
  }
  else PY_ERR_TYPE("argument must a be a number or dense matrix");
}

PyObject * matrix_sqrt(matrix *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A;

  if (!PyArg_ParseTuple(args, "O", &A)) return NULL;

  if (PyInt_Check(A) || PyFloat_Check(A)) {
    double f = PyFloat_AsDouble(A);
    if (f >= 0.0)
      return Py_BuildValue("d",sqrt(f));
    else PY_ERR(PyExc_ValueError, "domain error");
  }
  else if (PyComplex_Check(A)) {
    number n;
    convert_num[COMPLEX](&n, A, 1, 0);
    n.z = csqrt(n.z);
    return num2PyObject[COMPLEX](&n, 0);
  }
  else if (Matrix_Check(A) && (MAT_ID(A) == INT || MAT_ID(A) == DOUBLE)) {
    if (MAT_LGT(A) == 0)
      return (PyObject *)Matrix_New(MAT_NROWS(A),MAT_NCOLS(A),DOUBLE);

    double val = (MAT_ID(A) == INT ? MAT_BUFI(A)[0] : MAT_BUFD(A)[0]);

    int i;
    for (i=1; i<MAT_LGT(A); i++) {
      if (MAT_ID(A) == INT)
        val = MIN(val,(MAT_BUFI(A)[i]));
      else
        val = MIN(val,(MAT_BUFD(A)[i]));
    }

    if (val >= 0.0) {
      matrix *ret = Matrix_New(MAT_NROWS(A), MAT_NCOLS(A), DOUBLE);
      if (!ret) return PyErr_NoMemory();

      for (i=0; i<MAT_LGT(A); i++)
        MAT_BUFD(ret)[i] = sqrt((MAT_ID(A)== INT ?
            MAT_BUFI(A)[i] : MAT_BUFD(A)[i]));

      return (PyObject *)ret;

    }
    else PY_ERR(PyExc_ValueError, "domain error");
  }
  else if (Matrix_Check(A) && MAT_ID(A) == COMPLEX) {
    matrix *ret = Matrix_New(MAT_NROWS(A), MAT_NCOLS(A), COMPLEX);
    if (!ret) return PyErr_NoMemory();

    int i;
    for (i=0; i<MAT_LGT(A); i++)
      MAT_BUFZ(ret)[i] = csqrt(MAT_BUFZ(A)[i]);

    return (PyObject *)ret;
  }
  else PY_ERR_TYPE("argument must a be a number or dense matrix");
}

PyObject * matrix_cos(matrix *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A;

  if (!PyArg_ParseTuple(args, "O", &A)) return NULL;

  if (PyInt_Check(A) || PyFloat_Check(A))
    return Py_BuildValue("d",cos(PyFloat_AsDouble(A)));

  else if (PyComplex_Check(A)) {
    number n;
    convert_num[COMPLEX](&n, A, 1, 0);
    n.z = ccos(n.z);
    return num2PyObject[COMPLEX](&n, 0);
  }

  else if (Matrix_Check(A)) {
    matrix *ret = Matrix_New(MAT_NROWS(A),MAT_NCOLS(A),
        (MAT_ID(A) == COMPLEX ? COMPLEX : DOUBLE));
    if (!ret) return PyErr_NoMemory();

    int_t i;
    if (MAT_ID(ret) == DOUBLE)
      for (i=0; i<MAT_LGT(ret); i++)
        MAT_BUFD(ret)[i] = cos(MAT_ID(A) == DOUBLE ? MAT_BUFD(A)[i] :
        MAT_BUFI(A)[i]);
    else
      for (i=0; i<MAT_LGT(ret); i++)
        MAT_BUFZ(ret)[i] = ccos(MAT_BUFZ(A)[i]);

    return (PyObject *)ret;
  }
  else PY_ERR_TYPE("argument must a be a number or dense matrix");
}

PyObject * matrix_sin(matrix *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A;

  if (!PyArg_ParseTuple(args, "O", &A)) return NULL;

  if (PyInt_Check(A) || PyFloat_Check(A))
    return Py_BuildValue("d",sin(PyFloat_AsDouble(A)));

  else if (PyComplex_Check(A)) {
    number n;
    convert_num[COMPLEX](&n, A, 1, 0);
    n.z = csin(n.z);
    return num2PyObject[COMPLEX](&n, 0);
  }

  else if (Matrix_Check(A)) {
    matrix *ret = Matrix_New(MAT_NROWS(A),MAT_NCOLS(A),
        (MAT_ID(A) == COMPLEX ? COMPLEX : DOUBLE));
    if (!ret) return PyErr_NoMemory();

    int_t i;
    if (MAT_ID(ret) == DOUBLE)
      for (i=0; i<MAT_LGT(ret); i++)
        MAT_BUFD(ret)[i] = sin(MAT_ID(A) == DOUBLE ? MAT_BUFD(A)[i] :
        MAT_BUFI(A)[i]);
    else
      for (i=0; i<MAT_LGT(ret); i++)
        MAT_BUFZ(ret)[i] = csin(MAT_BUFZ(A)[i]);

    return (PyObject *)ret;
  }
  else PY_ERR_TYPE("argument must a be a number or dense matrix");
}


