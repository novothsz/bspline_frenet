# Spline operations: shift, crop, extrapolate, integrate.
# Cleaned from OMG-tools spline_extra.py
# Original: Copyright (C) 2016 Ruben Van Parys & Tim Mercy, KU Leuven (LGPL v3)

from .spline import BSpline
from .basis import BSplineBasis
from casadi import SX, MX, mtimes, Function, vertcat
import numpy as np


def make_basis(degree: int, knot_intervals: int) -> BSplineBasis:
    """Create a uniform B-spline basis on [0, 1]."""
    knots = np.r_[np.zeros(degree),
                  np.linspace(0, 1, knot_intervals + 1),
                  np.ones(degree)]
    return BSplineBasis(knots, degree)


def evalspline(s, x):
    """Evaluate a spline at a CasADi symbolic variable x."""
    Bl = s.basis
    coeffs = s.coeffs
    k = Bl.knots
    basis = [[]]
    for i in range(len(k) - 1):
        if i < Bl.degree + 1 and Bl.knots[0] == Bl.knots[i]:
            basis[-1].append((x >= Bl.knots[i]) * (x <= Bl.knots[i + 1]))
        else:
            basis[-1].append((x > Bl.knots[i]) * (x <= Bl.knots[i + 1]))
    for d in range(1, Bl.degree + 1):
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
    result = 0.
    for l in range(len(Bl)):
        result += coeffs[l] * basis[-1][l]
    return result


def running_integral(spline):
    """Compute the running integral of a spline."""
    basis = spline.basis
    coeffs = spline.coeffs
    knots = basis.knots
    degree = basis.degree

    knots_int = np.r_[knots[0], knots, knots[-1]]
    degree_int = degree + 1
    basis_int = BSplineBasis(knots_int, degree_int)
    coeffs_int = [0.]
    for i in range(len(basis_int) - 1):
        coeffs_int.append(
            coeffs_int[i] + (knots[degree + i + 1] - knots[i]) / float(degree_int) * coeffs[i])
    if isinstance(coeffs, (MX, SX)):
        coeffs_int = vertcat(*coeffs_int)
    else:
        coeffs_int = np.array(coeffs_int)
    return BSpline(basis_int, coeffs_int)


def definite_integral(spline, a, b):
    """Compute the definite integral of a spline over [a, b]."""
    spline_int = running_integral(spline)
    return evalspline(spline_int, b) - evalspline(spline_int, a)


def shift_spline(coeffs, t_shift, basis):
    """Extract spline piece in [t_shift, T] and re-express in equidistant basis."""
    n_knots = len(basis) - basis.degree + 1
    knots = basis.knots
    degree = basis.degree
    knots2 = np.r_[t_shift * np.ones(degree),
                   np.linspace(t_shift, knots[-1], n_knots),
                   knots[-1] * np.ones(degree)]
    basis2 = BSplineBasis(knots2, degree)
    T_tf = basis2.transform(basis)
    return basis2, T_tf.dot(coeffs)


def extrapolate(coeffs, t_extra, basis):
    """Extrapolate a spline by t_extra beyond its domain."""
    basis2, T = extrapolate_T(basis, t_extra)
    return basis2, T.dot(coeffs)


def extrapolate_T(basis, t_extra):
    """Create transformation matrix for extrapolation by t_extra."""
    knots = basis.knots
    deg = basis.degree
    N = len(basis)
    m = 1
    while knots[-deg - 2 - m] >= knots[-deg - 2]:
        m += 1
    knots2 = np.r_[knots[:-deg - 1], knots[-deg - 1] * np.ones(m),
                   (knots[-1] + t_extra) * np.ones(deg + 1)]
    basis2 = BSplineBasis(knots2, deg)
    A = np.zeros((deg + 1, deg + 1))
    B = np.zeros((deg + 1, deg + 1))
    if m < deg + 1:
        eval_points = basis.greville()[-(deg + 1 - m):]
        a = basis2.eval_basis(eval_points).toarray()[:, -(deg + 1 + m):-m]
        b = basis.eval_basis(eval_points).toarray()[:, -(deg + 1):]
        a1, a2 = a[:, :m], a[:, m:]
        b1, b2 = b[:, :m], b[:, m:]
        A[:(deg + 1 - m), -(deg + 1):-m] = a2
        B[:(deg + 1 - m), :m] = b1 - a1
        B[:(deg + 1 - m), m:] = b2
    else:
        A[0, -(deg + 1)] = 1.
        B[0, -1] = 1.
    A1, B1 = np.identity(deg + 1), np.identity(deg + 1)
    for i in range(1, deg + 1):
        A1_tmp = np.zeros((deg + 1 - i, deg + 1 - i + 1))
        B1_tmp = np.zeros((deg + 1 - i, deg + 1 - i + 1))
        for j in range(deg + 1 - i):
            B1_tmp[j, j] = -(deg + 1 - i) / (knots[j + N] - knots[j + N - deg - 1 + i])
            B1_tmp[j, j + 1] = (deg + 1 - i) / (knots[j + N] - knots[j + N - deg - 1 + i])
            A1_tmp[j, j] = -(deg + 1 - i) / (knots2[j + N + m] - knots2[j + N - deg - 1 + m + i])
            A1_tmp[j, j + 1] = (deg + 1 - i) / (knots2[j + N + m] - knots2[j + N - deg - 1 + m + i])
        A1, B1 = A1_tmp.dot(A1), B1_tmp.dot(B1)
        if i >= deg + 1 - m:
            b1 = B1[-1, :]
            a1 = A1[-(deg - i + 1), :]
            A[i, :] = a1
            B[i, :] = b1
    _T = np.linalg.solve(A, B)
    _T[abs(_T) < 1e-10] = 0.
    T = np.zeros((N + m, N))
    T[:N, :N] = np.eye(N)
    T[-(deg + 1):, -(deg + 1):] = _T
    return basis2, T


def shift_over_knot(coeffs, basis):
    """Shift spline horizon by one knot interval, extrapolating at the end."""
    basis, T = shiftoverknot_T(basis)
    return basis, T.dot(coeffs)


def shiftoverknot_T(basis):
    """Create transformation matrix for shifting over one knot interval."""
    knots = basis.knots
    deg = basis.degree
    m = 1
    while knots[-deg - 2 - m] >= knots[-deg - 2]:
        m += 1
    t_shift = knots[deg + 1] - knots[0]
    T = np.diag(np.ones(len(basis) - m), m)
    _T = np.eye(deg + 1)
    for k in range(deg):
        _t = np.zeros((deg + 1 + k + 1, deg + 1 + k))
        for j in range(deg + 1 + k + 1):
            if j >= deg + 1:
                _t[j, j - 1] = 1.
            elif j <= k:
                _t[j, j] = 1.
            else:
                _t[j, j - 1] = (knots[j + deg - k] - t_shift) / (knots[j + deg - k] - knots[j])
                _t[j, j] = (t_shift - knots[j]) / (knots[j + deg - k] - knots[j])
        _T = _t.dot(_T)
    T[:deg, :deg + 1] = _T[deg + 1:, :]
    basis, T_extr = extrapolate_T(basis, knots[-1] - knots[-deg - 2])
    T[-(deg + 1):, -(deg + 1):] = T_extr[-(deg + 1):, -(deg + 1):]
    return basis, T


def shift_knot1_fwd(cfs, basis, t_shift):
    """Shift the first knot forward by t_shift."""
    if isinstance(cfs, (SX, MX)):
        cfs_sym = MX.sym('cfs', cfs.shape)
        t_shift_sym = MX.sym('t_shift')
        T = shiftfirstknot_T(basis, t_shift_sym)
        cfs2_sym = mtimes(T, cfs_sym)
        fun = Function('fun', [cfs_sym, t_shift_sym], [cfs2_sym]).expand()
        return fun(cfs, t_shift)
    else:
        T = shiftfirstknot_T(basis, t_shift)
        return T.dot(cfs)


def shift_knot1_bwd(cfs, basis, t_shift):
    """Shift the first knot backward by t_shift."""
    if isinstance(cfs, (SX, MX)):
        cfs_sym = SX.sym('cfs', cfs.shape)
        t_shift_sym = SX.sym('t_shift')
        _, Tinv = shiftfirstknot_T(basis, t_shift_sym, inverse=True)
        cfs2_sym = mtimes(Tinv, cfs_sym)
        fun = Function('fun', [cfs_sym, t_shift_sym], [cfs2_sym]).expand()
        return fun(cfs, t_shift)
    else:
        _, Tinv = shiftfirstknot_T(basis, t_shift, inverse=True)
        return Tinv.dot(cfs)


def shiftfirstknot_T(basis, t_shift, inverse=False):
    """Create transformation matrix that shifts the first (degree+1) knots by t_shift."""
    knots, deg = basis.knots, basis.degree
    N = len(basis)
    if isinstance(t_shift, SX):
        typ, sym = SX, True
    elif isinstance(t_shift, MX):
        typ, sym = MX, True
    else:
        typ, sym = np, False
    _T = typ.eye(deg + 1)
    for k in range(deg + 1):
        _t = typ.zeros((deg + 1 + k + 1, deg + 1 + k))
        for j in range(deg + 1 + k + 1):
            if j >= deg + 1:
                _t[j, j - 1] = 1.
            elif j <= k:
                _t[j, j] = 1.
            else:
                _t[j, j - 1] = (knots[j + deg - k] - t_shift) / (knots[j + deg - k] - knots[j])
                _t[j, j] = (t_shift - knots[j]) / (knots[j + deg - k] - knots[j])
        _T = mtimes(_t, _T) if sym else _t.dot(_T)
    T = typ.eye(N)
    T[:deg + 1, :deg + 1] = _T[deg + 1:, :]
    if inverse:
        Tinv = typ.eye(len(basis))
        for i in range(deg, -1, -1):
            Tinv[i, i] = 1. / T[i, i]
            for j in range(deg, i, -1):
                Tinv[i, j] = (-1. / T[i, i]) * sum([T[i, k] * Tinv[k, j]
                                                     for k in range(i + 1, deg + 2)])
        return T, Tinv
    else:
        return T


def knot_insertion_T(basis, knots_to_insert):
    """Create transformation matrix for knot insertion."""
    N = len(basis)
    knots = basis.knots.tolist()
    degree = basis.degree
    T = np.eye(N)
    for knot in knots_to_insert:
        _T = np.zeros((N + 1, N))
        for j in range(N + 1):
            if knot <= knots[j]:
                w = 0.
            elif knots[j] < knot < knots[j + degree + 1 - 1]:
                w = (knot - knots[j]) / (knots[j + degree + 1 - 1] - knots[j])
            else:
                w = 1.
            if j != 0:
                _T[j, j - 1] = (1. - w)
            if j != N:
                _T[j, j] = w
        T = _T.dot(T)
        N += 1
        knots = sorted(knots + [knot])
    return T, knots


def get_interval_T(basis, min_value, max_value):
    """Create transformation matrix to extract a piece of the spline."""
    knots = basis.knots
    degree = basis.degree
    n_min = len(np.where(knots == min_value)[0])
    n_max = len(np.where(knots == max_value)[0])
    min_knots = [min_value] * (degree + 1 - n_min)
    max_knots = [max_value] * (degree + 1 - n_max)
    T, knots2 = knot_insertion_T(basis, min_knots + max_knots)
    jmin = np.searchsorted(knots2, min_value, side='left')
    jmax = np.searchsorted(knots2, max_value, side='right')
    return T[jmin:jmax - degree - 1, :], knots2[jmin:jmax]


def crop_spline(spline, min_value, max_value):
    """Crop a spline to [min_value, max_value]."""
    T, knots2 = get_interval_T(spline.basis, min_value, max_value)
    if isinstance(spline.coeffs, (SX, MX)):
        coeffs2 = mtimes(T, spline.coeffs)
    else:
        coeffs2 = T.dot(spline.coeffs)
    basis2 = BSplineBasis(knots2, spline.basis.degree)
    return BSpline(basis2, coeffs2)
