# Cleaned from OMG-tools spline.py
# Original: Copyright (C) 2016 Ruben Van Parys & Tim Mercy, KU Leuven (LGPL v3)
# Removed: Gurobi, cvxopt, Nurbs, TSpline, TensorBSpline

from casadi import vertcat
import functools
import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
import casadi as cas
import hashlib

NO_POINTS = 501


def _md5(data):
    m = hashlib.md5()
    m.update(data)
    return m.digest()


def _memoize(f):
    """Memoization decorator for basis evaluation.
    Caches results keyed by (self, hash(x)).
    Does NOT cache CasADi symbolic evaluations (they must be recomputed)."""
    class _MemoDict(dict):
        def __init__(self, func):
            self.f = func

        def __call__(self, *args):
            x = args[1]
            module = getattr(type(x), '__module__', '').split('.')[0]
            if module == 'casadi':
                # Symbolic -- never cache
                return self.f(*args)
            key = (args[0], _md5(np.atleast_1d(x)))
            if key not in self:
                self[key] = self.f(*args)
            return self[key]

        def __get__(self, obj, objtype):
            return functools.partial(self.__call__, obj)

    return _MemoDict(f)


def _cached_class(klass):
    """Cache class instances by constructor arguments."""
    cache = {}

    @functools.wraps(klass, assigned=('__name__', '__module__'), updated=())
    class _Decorated(klass):
        __doc__ = klass.__doc__

        def __new__(cls, *args, **kwds):
            key = (cls,) + tuple([_md5(np.atleast_1d(k)) for k in args]) + tuple(kwds.items())
            try:
                inst = cache.get(key, None)
            except TypeError:
                inst = key = None
            if inst is None:
                inst = klass(*args, **kwds)
                inst.__class__ = cls
                if key is not None:
                    cache[key] = inst
            return inst

        def __init__(self, *args, **kwds):
            pass

    return _Decorated


class CsrMatrixAlt(csr_matrix):
    """Subclass csr_matrix to overload dot for CasADi MX/SX types."""

    def __init__(self, *args, **kwargs):
        csr_matrix.__init__(self, *args, **kwargs)

    def dot(self, other):
        if isinstance(other, (cas.MX, cas.SX)):
            return cas.mtimes(cas.DM(csr_matrix(self)), other)
        else:
            try:
                return super(CsrMatrixAlt, self).dot(other)
            except:
                return np.dot(self.toarray(), other)


class Basis:
    """A generic spline basis with a knot sequence and degree."""

    def __init__(self, knots, degree):
        self.knots = np.array(knots)
        self.degree = degree
        self._x = np.linspace(knots[0], knots[-1], NO_POINTS)

    def __len__(self):
        return len(self.knots) - self.degree - 1

    def __call__(self, x):
        return self.eval_basis(x)

    def _ind(self, i, x):
        """Indicator function between knots[i] and knots[i + 1]."""
        if i < self.degree + 1 and self.knots[0] == self.knots[i]:
            return (x >= self.knots[i]) * (x <= self.knots[i + 1])
        return (x > self.knots[i]) * (x <= self.knots[i + 1])

    def _combine(self, other, degree):
        from collections import Counter
        c_self = Counter(self.knots)
        c_other = Counter(other.knots)
        breaks = set(self.knots).union(other.knots)
        multiplicity = [max(c_self.get(b, -np.inf) + degree - self.degree,
                            c_other.get(b, -np.inf) + degree - other.degree)
                        for b in breaks]
        knots = sum([[b] * m for b, m in zip(breaks, multiplicity)], [])
        return self.__class__(sorted(knots), degree)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            degree = max(self.degree, other.degree)
            return self._combine(other, degree)
        elif isinstance(other, (float, int)):
            return self
        else:
            raise TypeError("Not a basis error")

    __radd__ = __add__
    __sub__ = __add__
    __rsub__ = __sub__

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            degree = self.degree + other.degree
            return self._combine(other, degree)
        elif isinstance(other, (float, int)):
            return self
        else:
            raise TypeError("Not a basis error")

    __rmul__ = __mul__

    def __pow__(self, power):
        if isinstance(power, int):
            degree = power * self.degree
            return self._combine(self, degree)
        else:
            raise TypeError("Power must be integer")

    def __eq__(self, other):
        return (self.knots.shape == other.knots.shape
                and all(self.knots == other.knots)
                and self.degree == other.degree)

    def __hash__(self):
        sh = [hash(self.degree)] + [hash(e) for e in self.knots]
        r = sh[0]
        for e in sh[1:]:
            r = r ^ e
        return r

    def insert_knots(self, knots):
        unique_knots = np.setdiff1d(knots, self.knots)
        knots = np.sort(np.append(self.knots, unique_knots))
        return self.__class__(knots, self.degree)

    def greville(self):
        """Return the Greville abscissae of the basis."""
        return [1. / self.degree * sum(self.knots[k + 1:k + self.degree + 1])
                for k in range(len(self))]

    def scale(self, factor, shift=0):
        knots = self.knots * factor + shift
        return self.__class__(knots, self.degree)


@_cached_class
class BSplineBasis(Basis):
    """A numerical B-spline basis using the Cox-de Boor formula."""

    @_memoize
    def eval_basis(self, x):
        """Evaluate the B-spline basis functions at x."""
        module = getattr(type(x), '__module__', '').split('.')[0]
        is_casadi = (module == 'casadi')
        if not is_casadi:
            x = np.array(x)

        k = self.knots
        basis = [[self._ind(i, x) * 1.0 for i in range(len(k) - 1)]]
        for d in range(1, self.degree + 1):
            basis.append([])
            for i in range(len(k) - d - 1):
                b = 0 * x
                bottom = k[i + d] - k[i]
                if bottom != 0:
                    b = (x - k[i]) * basis[d - 1][i] / bottom
                bottom = k[i + d + 1] - k[i + 1]
                if bottom != 0:
                    b += (k[i + d + 1] - x) * basis[d - 1][i + 1] / bottom
                basis[-1].append(b)

        if is_casadi:
            return vertcat(*basis[-1])
        else:
            return CsrMatrixAlt(np.c_[basis[-1]].T)

    def derivative(self, o=1):
        """Returns (derivative_basis, transformation_matrix) for the o-th derivative.

        Uses eq. (16) in [de Boor, Chapter X, 2001].
        """
        B = self.__class__(self.knots[o:-o], self.degree - o)
        P = np.eye(len(self))
        knots = self.knots
        for i in range(o):
            knots = knots[1:-1]
            delta_knots = knots[self.degree - i:] - knots[:- self.degree + i]
            T = np.zeros((len(self) - 1 - i, len(self) - i))
            j = np.arange(len(self) - 1 - i)
            T[(j, j)] = -1. / delta_knots
            T[(j, j + 1)] = 1. / delta_knots
            P = (self.degree - i) * np.dot(T, P)
        return B, CsrMatrixAlt(P)

    def support(self):
        """Return a list of support intervals for each basis function."""
        return list(zip(
            self.knots[:-(self.degree + 1)],
            self.knots[(self.degree + 1):]
        ))

    def pairs(self, other):
        """Return which pairs remain when multiplying two bases."""
        def is_valid(a, b):
            return max(a[0], b[0]) < min(a[1], b[1])
        i_self = self.support()
        i_other = other.support()
        pairs = np.where([[is_valid(j, x) for x in i_other]
                          for j in i_self])
        S = np.zeros((len(self), len(self) * len(other)))
        return pairs, S

    def transform(self, other, TOL=1e-10):
        """Transformation matrix T such that self(x) = T @ other(x)."""
        b = self(self._x).toarray()
        m = np.argmax(b, axis=0)
        xmax = self._x[m]
        if isinstance(other, BSplineBasis):
            T = la.solve(b[m, :], other(xmax).toarray())
        else:
            try:
                T = la.solve(b[m, :], other(xmax))
            except:
                T = la.solve(b[m, :], other(m))
        T[abs(T) < TOL] = 0.
        return CsrMatrixAlt(T)
