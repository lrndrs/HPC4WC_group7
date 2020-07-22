# ******************************************************
#      Script: Numba Stencils
#      Author: HPC4WC Group 7
#        Date: 02.07.2020
# ******************************************************

import numpy as np
from numba import jit  # , njit


def laplacian1d(in_field, tmp_field, num_halo=1, extend=0):
    """Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    num_halo  -- number of halo points
    
    Keyword arguments:
    extend    -- extend computation into halo-zone by this number of points
    """

    I, J, K = in_field.shape

    ib = np.int64(num_halo - extend)
    ie = np.int64(-num_halo + extend)

    for i in range(ib, I + ie):
        tmp_field[i, :, :] = (
            -2.0 * in_field[i, :, :] + in_field[i - 1, :, :] + in_field[i + 1, :, :]
        )

    return tmp_field


def laplacian2d(in_field, tmp_field, num_halo=1, extend=0):
    """Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    num_halo  -- number of halo points
    
    Keyword arguments:
    extend    -- extend computation into halo-zone by this number of points
    """

    I, J, K = in_field.shape

    ib = np.int64(num_halo - extend)
    ie = np.int64(-num_halo + extend)
    jb = np.int64(num_halo - extend)
    je = np.int64(-num_halo + extend)

    for i in range(ib, I + ie):
        for j in range(jb, J + je):

            tmp_field[i, j, :] = (
                -4.0 * in_field[i, j, :]
                + in_field[i - 1, j, :]
                + in_field[i + 1, j, :]
                + in_field[i, j - 1, :]
                + in_field[i, j + 1, :]
            )

    return tmp_field


def laplacian3d(in_field, tmp_field, num_halo=1, extend=0):
    """Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    num_halo  -- number of halo points
    
    Keyword arguments:
    extend    -- extend computation into halo-zone by this number of points
    """

    I, J, K = in_field.shape

    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend
    kb = num_halo - extend
    ke = -num_halo + extend

    for i in range(ib, I + ie):
        for j in range(jb, J + je):
            for k in range(kb, K + ke):
                tmp_field[i, j, k] = (
                    -6.0 * in_field[i, j, k]
                    + in_field[i - 1, j, k]
                    + in_field[i + 1, j, k]
                    + in_field[i, j - 1, k]
                    + in_field[i, j + 1, k]
                    + in_field[i, j, k - 1]
                    + in_field[i, j, k + 1]
                )

    return tmp_field
