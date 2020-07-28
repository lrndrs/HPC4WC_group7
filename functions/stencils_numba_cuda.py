# ******************************************************
#      Script: Numba Loop
#      Author: HPC4WC Group 7
#        Date: 02.07.2020
# ******************************************************

import numpy as np
from numba import cuda


@cuda.jit
def increment_by_one(an_array,another_array):
    pos = cuda.grid(1)
    if pos < an_array.size:
        another_array[pos] = an_array[pos]



@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1
        

@cuda.jit
def increment_a_3D_array(an_array):
    x, y, z = cuda.grid(3)
    if x < an_array.shape[0] and y < an_array.shape[1] and z < an_array.shape[2]:
       an_array[x, y, z] += 1
        
        
@cuda.jit
def test(in_field,out_field):
    """
    Simple test function that returns a copy of the in_field.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    out_field : a copy of the in_field.
    
    """
    x, y, z = cuda.grid(3)
    if x < in_field.shape[0] and y < in_field.shape[1] and z < in_field.shape[2]:
       out_field[x,y,z] = in_field[x,y,z] 



def laplacian1d(in_field, out_field, num_halo=1):
    """Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian computed in i-direction.
    
    """

    I, J, K = in_field.shape

    for i in range(num_halo, I - num_halo):
        for j in range(num_halo, J - num_halo):
            for k in range(num_halo, K - num_halo):
                out_field[i, j, k] = (
                    -2.0 * in_field[i, j, k]
                    + in_field[i - 1, j, k]
                    + in_field[i + 1, j, k]
                )

    return out_field


def laplacian2d(in_field, out_field, num_halo=1):
    """
    Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian computed in i- and j-direction (horizontal Laplacian).
    
    """
    I, J, K = in_field.shape

    for i in range(num_halo, I - num_halo):
        for j in range(num_halo, J - num_halo):
            for k in range(num_halo, K - num_halo):
                out_field[i, j, k] = (
                    -4.0 * in_field[i, j, k]
                    + in_field[i - 1, j, k]
                    + in_field[i + 1, j, k]
                    + in_field[i, j - 1, k]
                    + in_field[i, j + 1, k]
                )

    return out_field


def laplacian3d(in_field, out_field, num_halo=1):
    """
    Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian computed in i-, j- and k- direction (full Laplacian).
    
    """

    I, J, K = in_field.shape

    for i in range(num_halo, I - num_halo):
        for j in range(num_halo, J - num_halo):
            for k in range(num_halo, K - num_halo):
                out_field[i, j, k] = (
                    -6.0 * in_field[i, j, k]
                    + in_field[i - 1, j, k]
                    + in_field[i + 1, j, k]
                    + in_field[i, j - 1, k]
                    + in_field[i, j + 1, k]
                    + in_field[i, j, k - 1]
                    + in_field[i, j, k + 1]
                )

    return out_field


def FMA(in_field, in_field2, in_field3, out_field, num_halo=0):
    """
    Pointwise stencil to test for fused multiply-add with an explicit nested loop in numba.
    
    Parameters
    ----------
    in_field,in_field2, in_field3  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : fused multiply-add applied to in_field.
    
    """

    I, J, K = in_field.shape

    for i in range(num_halo, I - num_halo):
        for j in range(num_halo, J - num_halo):
            for k in range(num_halo, K - num_halo):

                out_field[i, j, k] = (
                    in_field[i, j, k] + in_field2[i, j, k] * in_field3[i, j, k]
                )

    return out_field


def lapoflap1d(in_field, tmp_field, out_field, num_halo=2):
    """
    Compute Laplacian in i-direction using 2nd-order centered differences with an explicit nested loop in numba.
    
    Parameters
    ----------
    in_field   : input field (nx x ny x nz).
    tmp_field  : intermediate result (must be of same size as in_field).
    out_field : result (must be of same size as in_field).
    num_halo   : number of halo points.
    
    Returns
    -------
    out_field  : in_field with Laplacian of the Laplacian computed in i-direction.
    
    """

    I, J, K = in_field.shape

    for i in range(num_halo - 1, I - num_halo + 1):
        for j in range(num_halo - 1, J - num_halo + 1):
            for k in range(num_halo - 1, K - num_halo + 1):
                tmp_field[i, j, k] = (
                    -2.0 * in_field[i, j, k]
                    + in_field[i - 1, j, k]
                    + in_field[i + 1, j, k]
                )

    for i in range(num_halo, I - num_halo):
        for j in range(num_halo, J - num_halo):
            for k in range(num_halo, K - num_halo):
                out_field[i, j, k] = (
                    -2.0 * tmp_field[i, j, k]
                    + tmp_field[i - 1, j, k]
                    + tmp_field[i + 1, j, k]
                )

    return out_field


def lapoflap2d(in_field, tmp_field, out_field, num_halo=2):
    """
    Compute Laplacian of the Laplacian in i and j-direction using 2nd-order centered differences with an explicit nested loop in numba.
    
    Parameters
    ----------
    in_field   : input field (nx x ny x nz).
    tmp_field  : intermediate result (must be of same size as in_field).
    out_field : result (must be of same size as in_field).
    num_halo   : number of halo points.
    
    Returns
    -------
    out_field  : in_field with Laplacian of the Laplacian computed in i- and j-direction (horizontally).
    
    """

    I, J, K = in_field.shape

    for i in range(num_halo - 1, I - num_halo + 1):
        for j in range(num_halo - 1, J - num_halo + 1):
            for k in range(num_halo - 1, K - num_halo + 1):
                tmp_field[i, j, k] = (
                    -4.0 * in_field[i, j, k]
                    + in_field[i - 1, j, k]
                    + in_field[i + 1, j, k]
                    + in_field[i, j - 1, k]
                    + in_field[i, j + 1, k]
                )

    for i in range(num_halo, I - num_halo):
        for j in range(num_halo, J - num_halo):
            for k in range(num_halo, K - num_halo):
                out_field[i, j, k] = (
                    -4.0 * tmp_field[i, j, k]
                    + tmp_field[i - 1, j, k]
                    + tmp_field[i + 1, j, k]
                    + tmp_field[i, j - 1, k]
                    + tmp_field[i, j + 1, k]
                )

    return out_field


def lapoflap3d(in_field, tmp_field, out_field, num_halo=2):
    """
    Compute Laplacian of the Laplacian in i,j,k-direction using 2nd-order centered differences with an explicit nested loop in numba.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    tmp_field  : intermediate result (must be of same size as in_field).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian of the Laplacian computed in i-, j- and k- direction.
    
    """
    I, J, K = in_field.shape

    for i in range(num_halo - 1, I - num_halo + 1):
        for j in range(num_halo - 1, J - num_halo + 1):
            for k in range(num_halo - 1, K - num_halo + 1):
                tmp_field[i, j, k] = (
                    -6.0 * in_field[i, j, k]
                    + in_field[i - 1, j, k]
                    + in_field[i + 1, j, k]
                    + in_field[i, j - 1, k]
                    + in_field[i, j + 1, k]
                    + in_field[i, j, k - 1]
                    + in_field[i, j, k + 1]
                )

    for i in range(num_halo, I - num_halo):
        for j in range(num_halo, J - num_halo):
            for k in range(num_halo, K - num_halo):
                out_field[i, j, k] = (
                    -6.0 * tmp_field[i, j, k]
                    + tmp_field[i - 1, j, k]
                    + tmp_field[i + 1, j, k]
                    + tmp_field[i, j - 1, k]
                    + tmp_field[i, j + 1, k]
                    + tmp_field[i, j, k - 1]
                    + tmp_field[i, j, k + 1]
                )

    return out_field
