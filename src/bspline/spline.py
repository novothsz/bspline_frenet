# Cleaned from OMG-tools spline.py
# Original: Copyright (C) 2016 Ruben Van Parys & Tim Mercy, KU Leuven (LGPL v3)
# Removed: Gurobi branches, cvxopt branches, Nurbs, __div__

import numpy as np
import casadi as cas
from .basis import BSplineBasis, CsrMatrixAlt


class Spline:
    """Base spline class: a basis + coefficients."""

    def __init__(self, basis, coeffs):
        self.coeffs = coeffs
        self.basis = basis

    def __call__(self, x):
        try:
            return self.basis(x).dot(self.coeffs)
        except:
            try:
                return cas.dot(self.basis(x), self.coeffs)
            except:
                return cas.dot(cas.vertcat(*self.basis(x)[0]), self.coeffs)

    def __len__(self):
        return len(self.basis)

    def __eq__(self, other):
        return (self.basis == other.basis
                and type(self.coeffs) == type(other.coeffs)
                and all(self.coeffs == other.coeffs))

    def __hash__(self):
        sh = [hash(self.basis)] + [hash(e) for e in self.coeffs]
        r = sh[0]
        for e in sh[1:]:
            r = r ^ e
        return r


class BSpline(Spline):
    """B-spline curve: a BSplineBasis + coefficients.

    Supports CasADi MX/SX coefficients for symbolic optimization
    and numpy coefficients for numerical evaluation.
    """

    def __add__(self, other):
        if isinstance(other, self.__class__):
            basis = self.basis + other.basis
            coeffs = (basis.transform(self.basis).dot(self.coeffs) +
                      basis.transform(other.basis).dot(other.coeffs))
        else:
            try:
                basis = self.basis
                coeffs = self.coeffs + other
            except:
                raise NotImplementedError("Incompatible datatype")
        return self.__class__(basis, coeffs)

    __radd__ = __add__

    def __neg__(self):
        return self.__class__(self.basis, -self.coeffs)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            basis = self.basis * other.basis
            pairs, S = self.basis.pairs(other.basis)
            b_self = self.basis(basis._x)
            b_other = other.basis(basis._x)
            basis_product = b_self[:, pairs[0]].multiply(b_other[:, pairs[1]])
            T = basis.transform(lambda y: basis_product.toarray()[y, :])

            coeffs_product = (self.coeffs[pairs[0].tolist()] *
                              other.coeffs[pairs[1].tolist()])
            return self.__class__(basis, T.dot(coeffs_product))
        else:
            try:
                basis = self.basis
                coeffs = other * self.coeffs
                return self.__class__(basis, coeffs)
            except:
                raise NotImplementedError("Incompatible datatype")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        if isinstance(power, int):
            a = self
            for i in range(1, power):
                a *= self
            return a
        else:
            raise TypeError("Exponent must be integer")

    def derivative(self, o=1):
        """Return the o-th derivative as a new BSpline."""
        if o == 0:
            return self
        Bd, Pd = self.basis.derivative(o=o)
        return self.__class__(Bd, Pd.dot(self.coeffs))

    def insert_knots(self, knots):
        """Return an equivalent spline with additional knots."""
        basis = self.basis.insert_knots(knots)
        coeffs = basis.transform(self.basis).dot(self.coeffs)
        return self.__class__(basis, coeffs)

    def integral(self):
        """Return the value of the integral over the support."""
        knots = self.basis.knots
        coeffs = self.coeffs
        d = self.basis.degree
        K = CsrMatrixAlt(np.diag((knots[d + 1:] - knots[:-(d + 1)]) / (d + 1)))
        return sum(K.dot(coeffs))

    def scale(self, factor, shift=0):
        """Scale and shift the domain of the spline."""
        basis = self.basis.scale(factor, shift=shift)
        return self.__class__(basis, self.coeffs)
