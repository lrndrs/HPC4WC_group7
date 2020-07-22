import numpy as np
from functions.halo_functions import update_halo, add_halo_points


def test(in_field):
    """
    Simple test function that returns a copy of the in_field.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    tmp_field : a copy of the in_field.
    
    """
    tmp_field = np.copy(in_field)

    return tmp_field


def laplacian1d(in_field, tmp_field, num_halo=1, extend=0):
    """
    Compute Laplacian in i-direction using 2nd-order centered differences.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    tmp_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    extend    : extend computation into halo-zone by this number of points.
    
    Returns
    -------
    tmp_field : in_field with Laplacian computed in i-direction.
    
    """

    ib = num_halo - extend
    ie = -num_halo + extend

    tmp_field[ib:ie, :, :] = (
        -2.0 * in_field[ib:ie, :, :]
        + in_field[ib - 1 : ie - 1, :, :]
        + in_field[ib + 1 : ie + 1 if ie != -1 else None, :, :]
    )

    return tmp_field


def laplacian2d(in_field, tmp_field, num_halo=1, extend=0):
    """
    Compute Laplacian in i- and j-direction using 2nd-order centered differences.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    tmp_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    extend    : extend computation into halo-zone by this number of points.
    
    Returns
    -------
    tmp_field : in_field with Laplacian computed in i- and j-direction (horizontal Laplacian).
    
    """

    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend

    tmp_field[ib:ie, jb:je, :] = (
        -4.0 * in_field[ib:ie, jb:je, :]
        + in_field[ib - 1 : ie - 1, jb:je, :]
        + in_field[ib + 1 : ie + 1 if ie != -1 else None, jb:je, :]
        + in_field[ib:ie, jb - 1 : je - 1, :]
        + in_field[ib:ie, jb + 1 : je + 1 if je != -1 else None, :]
    )

    return tmp_field


def laplacian3d(in_field, tmp_field, num_halo=1, extend=0):
    """
    Compute Laplacian in i-, j- and k-direction using 2nd-order centered differences.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    tmp_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    extend    : extend computation into halo-zone by this number of points.
    
    Returns
    -------
    tmp_field : in_field with Laplacian computed in i-, j- and k- direction (full Laplacian).
    
    """

    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend
    kb = num_halo - extend
    ke = -num_halo + extend

    tmp_field[ib:ie, jb:je, kb:ke] = (
        -6.0 * in_field[ib:ie, jb:je, kb:ke]
        + in_field[ib - 1 : ie - 1, jb:je, kb:ke]
        + in_field[ib + 1 : ie + 1 if ie != -1 else None, jb:je, kb:ke]
        + in_field[ib:ie, jb - 1 : je - 1, kb:ke]
        + in_field[ib:ie, jb + 1 : je + 1 if je != -1 else None, kb:ke]
        + in_field[ib:ie, jb:je, kb - 1 : ke - 1]
        + in_field[ib:ie, jb:je, kb + 1 : ke + 1 if ke != -1 else None]
    )

    return tmp_field


def FMA(in_field, in_field2, in_field3, tmp_field, num_halo=0, extend=0):
    """
    Pointwise stencil to test for fused multiply-add 
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    tmp_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    extend    : extend computation into halo-zone by this number of points.
    
    Returns
    -------
    tmp_field : fused multiply-add applied to in_field.
    
    """

    tmp_field = in_field + in_field2 * in_field3

    return tmp_field


def lapoflap1d(in_field, tmp_field, tmp2_field, num_halo=2, extend=1):
    """
    Compute Laplacian of the Laplacian in i-direction using 2nd-order centered differences.
    
    Parameters
    ----------
    in_field   : input field (nx x ny x nz).
    tmp_field  : intermediate result (must be of same size as in_field).
    tmp2_field : result (must be of same size as in_field).
    num_halo   : number of halo points.
    extend     : extend computation into halo-zone by this number of points.
    
    Returns
    -------
    tmp2_field  : in_field with Laplacian of the Laplacian computed in i-direction.
    
    """

    ib = num_halo - extend
    ie = -num_halo + extend
    tmp_field[ib:ie, :, :] = (
        -2.0 * in_field[ib:ie, :, :]
        + in_field[ib - 1 : ie - 1, :, :]
        + in_field[ib + 1 :, :, :]
    )

    extend = 0
    ib = num_halo - extend
    ie = -num_halo + extend
    tmp2_field[ib:ie, :, :] = (
        -2.0 * tmp_field[ib:ie, :, :]
        + tmp_field[ib - 1 : ie - 1, :, :]
        + tmp_field[ib + 1 : ie + 1, :, :]
    )

    return tmp2_field


def lapoflap2d(in_field, tmp_field, tmp2_field, num_halo=2, extend=1):
    """
    Compute Laplacian of the Laplacian in i- and j-direction using 2nd-order centered differences.
    
    Parameters
    ----------
    in_field   : input field (nx x ny x nz).
    tmp_field  : intermediate result (must be of same size as in_field).
    tmp2_field : result (must be of same size as in_field).
    num_halo   : number of halo points.
    extend     : extend computation into halo-zone by this number of points.
    
    Returns
    -------
    tmp2_field  : in_field with Laplacian of the Laplacian computed in i- and j-direction (horizontally).
    
    """
    ib = num_halo - extend #1
    ie = -num_halo + extend #-1
    jb = num_halo - extend #1
    je = -num_halo + extend #-1
    tmp_field[ib:ie, jb:je, :] = (
        -4.0 * in_field[ib:ie, jb:je, :]
        + in_field[ib - 1 : ie - 1, jb:je, :]
        + in_field[ib + 1 : , jb:je, :]
        + in_field[ib:ie, jb - 1 : je - 1, :]
        + in_field[ib:ie, jb + 1 : , :]
    )

    extend = 0
    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend
    tmp2_field[ib:ie, jb:je, :] = (
        -4.0 * tmp_field[ib:ie, jb:je, :]
        + tmp_field[ib - 1 : ie - 1, jb:je, :]
        + tmp_field[ib + 1 : -1, jb:je, :]
        + tmp_field[ib:ie, jb - 1 : je - 1, :]
        + tmp_field[ib:ie, jb + 1 : -1, :]
    )

    return tmp2_field


def lapoflap3d(in_field, tmp_field, tmp2_field, num_halo=2, extend=1):
    """
    Compute Laplacian of the Laplacian in i-, j- and k-direction using 2nd-order centered differences.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    tmp_field  : intermediate result (must be of same size as in_field).
    tmp2_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    extend    : extend computation into halo-zone by this number of points.
    
    Returns
    -------
    tmp2_field : in_field with Laplacian of the Laplacian computed in i-, j- and k- direction.
    
    """

    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend
    kb = num_halo - extend
    ke = -num_halo + extend
    tmp_field[ib:ie, jb:je, kb:ke] = (
        -6.0 * in_field[ib:ie, jb:je, kb:ke]
        + in_field[ib - 1 : ie - 1, jb:je, kb:ke]
        + in_field[ib + 1 : , jb:je, kb:ke]
        + in_field[ib:ie, jb - 1 : je - 1, kb:ke]
        + in_field[ib:ie, jb + 1 : , kb:ke]
        + in_field[ib:ie, jb:je, kb - 1 : ke - 1]
        + in_field[ib:ie, jb:je, kb + 1 : ]
    )

    extend = 0
    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend
    kb = num_halo - extend
    ke = -num_halo + extend

    tmp2_field[ib:ie, jb:je, kb:ke] = (
        -6.0 * tmp_field[ib:ie, jb:je, kb:ke]
        + tmp_field[ib - 1 : ie - 1, jb:je, kb:ke]
        + tmp_field[ib + 1 : -1, jb:je, kb:ke]
        + tmp_field[ib:ie, jb - 1 : je - 1, kb:ke]
        + tmp_field[ib:ie, jb + 1 : -1, kb:ke]
        + tmp_field[ib:ie, jb:je, kb - 1 : ke - 1]
        + tmp_field[ib:ie, jb:je, kb + 1 : -1]
    )

    return tmp2_field
