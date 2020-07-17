# ******************************************************
#      Script: Numba Stencils
#      Author: HPC4WC Group 7
#        Date: 02.07.2020
# ******************************************************

import numpy as np
from numba import jit, njit, vectorize, stencil, stencils
from numba import vectorize, guvectorize, float64, int32


@jit(nopython=True)
def test_numba(in_field):
    # simple test function that does nothing
    out_field = np.copy(in_field)

    return out_field


@jit(nopython=True)
def laplacian_numba(in_field, lap_field, dim_stencil, num_halo=1, extend=0):
    """Compute Laplacian using 2nd-order centered differences.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    dim_stencil       -- number of dimensions of the stencil (1-3)
    num_halo  -- number of halo points
    
    Keyword arguments:
    extend    -- extend computation into halo-zone by this number of points
    """

    # since laplacian is not defined for pointwise stencils.
    assert (
        0 < dim_stencil <= 3
    ), "The laplacian is not defined for pointwise stencils. Please choose between 1 to 3 dimensions."

    if dim_stencil == 1:
        ib = num_halo - extend
        ie = -num_halo + extend

        # lap_field[ib:ie] = - 2. * in_field[ib:ie]  \
        #    + in_field[ib - 1:ie - 1] + in_field[ib + 1:ie + 1 if ie != -1 else None]
        lap_field[:, :, ib:ie] = (
            -2.0 * in_field[:, :, ib:ie]
            + in_field[:, :, ib - 1 : ie - 1]
            + in_field[:, :, ib + 1 : ie + 1 if ie != -1 else None]
        )

        return lap_field

    if dim_stencil == 2:
        ib = num_halo - extend
        ie = -num_halo + extend
        jb = num_halo - extend
        je = -num_halo + extend

        # lap_field[jb:je, ib:ie] = - 4. * in_field[jb:je, ib:ie]  \
        #    + in_field[jb:je, ib - 1:ie - 1] + in_field[jb:je, ib + 1:ie + 1 if ie != -1 else None]  \
        #    + in_field[jb - 1:je - 1, ib:ie] + in_field[jb + 1:je + 1 if je != -1 else None, ib:ie]
        lap_field[:, jb:je, ib:ie] = (
            -4.0 * in_field[:, jb:je, ib:ie]
            + in_field[:, jb:je, ib - 1 : ie - 1]
            + in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
            + in_field[:, jb - 1 : je - 1, ib:ie]
            + in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]
        )

        return lap_field

    if dim_stencil == 3:
        ib = num_halo - extend
        ie = -num_halo + extend
        jb = num_halo - extend
        je = -num_halo + extend
        kb = num_halo - extend
        ke = -num_halo + extend

        lap_field[kb:ke, jb:je, ib:ie] = (
            -6.0 * in_field[kb:ke, jb:je, ib:ie]
            + in_field[kb:ke, jb:je, ib - 1 : ie - 1]
            + in_field[kb:ke, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
            + in_field[kb:ke, jb - 1 : je - 1, ib:ie]
            + in_field[kb:ke, jb + 1 : je + 1 if je != -1 else None, ib:ie]
            + in_field[kb - 1 : ke - 1, jb:je, ib:ie]
            + in_field[kb + 1 : ke + 1 if ke != -1 else None, jb:je, ib:ie]
        )

        return lap_field


@jit(nopython=True, debug=True)
# @njit()
def laplacian_numbaloop(in_field, lap_field, dim_stencil, num_halo=1, extend=0):
    """Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    dim_stencil       -- number of dimensions of the stencil (1-3)
    num_halo  -- number of halo points
    
    Keyword arguments:
    extend    -- extend computation into halo-zone by this number of points
    """

    # since laplacian is not defined for pointwise stencils.
    assert (
        0 < dim_stencil <= 3
    ), "The laplacian is not defined for pointwise stencils. Please choose between 1 to 3 dimensions."

    lap_field = np.empty_like(in_field)
    I, J, K = in_field.shape

    if dim_stencil == 1:
        kb = np.int64(num_halo - extend)
        ke = np.int64(-num_halo + extend)

        # lap_field[ib:ie] = - 2. * in_field[ib:ie]  \
        #    + in_field[ib - 1:ie - 1] + in_field[ib + 1:ie + 1 if ie != -1 else None]
        # lap_field[:, :, ib:ie] = - 2. * in_field[:, :, ib:ie]  \
        #   + in_field[:, :, ib - 1:ie - 1] + in_field[:, :, ib + 1:ie + 1 if ie != -1 else None]

        # for i in range(I):
        # for j in range(J):
        #    lap_field[i,j,0]=-2 * in_field[i,j,0]
        for k in range(kb, K + ke):
            lap_field[:, :, k] = (
                -2.0 * in_field[:, :, k] + in_field[:, :, k - 1] + in_field[:, :, k + 1]
            )  # + in_field[i, j, k + 1 if ie != -1 else None]

        return lap_field

    if dim_stencil == 2:
        kb = np.int64(num_halo - extend)
        ke = np.int64(-num_halo + extend)
        jb = np.int64(num_halo - extend)
        je = np.int64(-num_halo + extend)

        # lap_field[jb:je, ib:ie] = - 4. * in_field[jb:je, ib:ie]  \
        #    + in_field[jb:je, ib - 1:ie - 1] + in_field[jb:je, ib + 1:ie + 1 if ie != -1 else None]  \
        #    + in_field[jb - 1:je - 1, ib:ie] + in_field[jb + 1:je + 1 if je != -1 else None, ib:ie]
        # lap_field[:, jb:je, ib:ie] = - 4. * in_field[:, jb:je, ib:ie]  \
        #     + in_field[:, jb:je, ib - 1:ie - 1] + in_field[:, jb:je, ib + 1:ie + 1]  \
        #     + in_field[:, jb - 1:je - 1, ib:ie] + in_field[:, jb + 1:je + 1, ib:ie]

        # for i in range(I):
        for j in range(jb, J + je):
            for k in range(kb, K + ke):
                lap_field[:, j, k] = (
                    -4.0 * in_field[:, j, k]
                    + in_field[:, j, k - 1]
                    + in_field[:, j, k + 1]
                    + in_field[:, j - 1, k]
                    + in_field[:, j + 1, k]
                )  # + in_field[i, j, k + 1 if ie != -1 else None]

        return lap_field

    if dim_stencil == 3:
        ib = num_halo - extend
        ie = -num_halo + extend
        jb = num_halo - extend
        je = -num_halo + extend
        kb = num_halo - extend
        ke = -num_halo + extend

        # lap_field[kb:ke, jb:je, ib:ie] = - 6. * in_field[kb:ke, jb:je, ib:ie]  \
        #     + in_field[kb:ke, jb:je, ib - 1:ie - 1] + in_field[kb:ke, jb:je, ib + 1:ie + 1 if ie != -1 else None]  \
        #     + in_field[kb:ke, jb - 1:je - 1, ib:ie] + in_field[kb:ke, jb + 1:je + 1 if je != -1 else None, ib:ie]  \
        #     + in_field[kb - 1:ke - 1, jb:je, ib:ie] + in_field[kb + 1:ke  + 1 if ke != -1 else None, jb:je , ib:ie]

        for i in range(ib, I + ie):
            for j in range(jb, J + je):
                for k in range(kb, K + ke):
                    lap_field[i, j, k] = (
                        -6.0 * in_field[i, j, k]
                        + in_field[i, j, k - 1]
                        + in_field[i, j, k + 1]
                        + in_field[i, j - 1, k]
                        + in_field[i, j + 1, k]
                        + in_field[i - 1, j, k]
                        + in_field[i + 1, j, k]
                    )

        return lap_field


@jit(nopython=True)
def FMA_numba(in_field, dim_stencil=0, num_halo=0, extend=0):
    """pointwise stencil to test for fused multiply-add 
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    tmp_field -- result (must be same size as in_field)
    dim_stencil       -- number of dimension (0)
    num_halo  -- number of halo points
    
    Keyword arguments:
    extend    -- extend computation into halo-zone by this number of points
    """

    # FMA is a pointwise stencil
    assert (
        0 == dim_stencil
    ), "Please do not provide any input for dim_stencil for this stencil or set dim_stencil=0."

    # not finished; fields should likely be initialized in main.py since the initialisation also takes time.
    in_field2 = np.ones_like(in_field)
    in_field3 = np.ones_like(in_field) * 4.2
    # get_random_field(dim, nx+2*num_halo, ny+2*num_halo, nz+2*num_halo)
    # in_field3_FMA = get_random_field(dim, nx+2*num_halo, ny+2*num_halo, nz+2*num_halo)

    FMA_field = in_field + in_field2 * in_field3
    return FMA_field


@stencil
def laplacian1d_numbastencil_help(in_field):
    """
    Numpy function that computes the Laplacian of the in_field in i-direction. This function can be boosted with the @njit decorater as implemented in the function laplacian1d_numbastencil.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    
    """
    return -2.0 * in_field[0, 0, 0] + in_field[-1, 0, 0] + in_field[+1, 0, 0]


@njit()
def laplacian1d_numbastencil(in_field):
    """
    Function that boosts the function laplacian1d_numbastencil_help
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    
    """
    return laplacian1d_numbastencil_help(in_field)


@stencil
def laplacian2d_numbastencil(in_field):
    """
    Numpy function that computes the Laplacian of the in_field in i- and j-direction. This function can be boosted with the @njit decorater as implemented in the function laplacian2d_numbastencil.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    
    """
    return (
        -4.0 * in_field[0, 0, 0]
        + in_field[-1, 0, 0]
        + in_field[1, 0, 0]
        + in_field[0, -1, 0]
        + in_field[0, +1, 0]
    )


@njit()
def laplacian2d_numbastencil_help(in_field):
    """
    Function that boosts the function laplacian2d_numbastencil_help
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    
    """
    return laplacian2d_numbastencil_help(in_field)


@stencil
def laplacian3d_numbastencil_help(in_field):
    """
    Numpy function that computes the Laplacian of the in_field in i-, j- and k-direction. This function can be boosted with the @njit decorater as implemented in the function laplacian3d_numbastencil.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    
    """
    return (
        -6.0 * in_field[0, 0, 0]
        + in_field[-1, 0, 0]
        + in_field[+1, 0, 0]
        + in_field[0, -1, 0]
        + in_field[0, +1, 0]
        + in_field[0, 0, -1]
        + in_field[0, 0, +1]
    )


@njit()
def laplacian3d_numbastencil(in_field):
    """
    Function that boosts the function laplacian3d_numbastencil_help
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    
    """
    return laplacian3d_numbastencil_help(in_field)


# vectorize only works for point-wise stencils. maybe @guvectorize does the job for non-point-wise stencils
# did not yet found a way to deal with the halo
@vectorize([float64(float64, float64, float64)], nopython=True)  # target="parallel"
def FMA_numbavectorize(in_field, in_field2, in_field3):
    return in_field + in_field2 * in_field3


# @vectorize([float64(float64)])
# def laplacian1d_numbavectorize(in_field):
#    return - 2. * in_field[0, 0, 0]  \
#        + in_field[- 1, 0, 0] + in_field[+ 1 , 0, 0]

# @guvectorize([(float64[:], float64[:], float64, float64)], '(n),(),()->(n)')
# def laplacian1d_numbaguvectorize(in_field, tmp_field, num_halo=1, extend=0 ):
#    for i in range(num_halo -extend, in_field.shape[0] - num_halo + extend):
#        tmp_field[i, : , : ] = - 2. * in_field[i, 0, 0]  \
#        + in_field[i - 1, 0, 0] + in_field[i + 1 , 0, 0]
