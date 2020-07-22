# ******************************************************
#      Script: Numba Loop
#      Author: HPC4WC Group 7
#        Date: 02.07.2020
# ******************************************************

import numpy as np


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



def laplacian1d(in_field, tmp_field):
    """Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    in_field  -- input field (nx x ny x nz with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    
    """

    I, J, K = in_field.shape

    for i in range(1, I -1):
        tmp_field[i,:, :] = (
            -2.0 * in_field[i, :, :] + in_field[i - 1, :, :] + in_field[i + 1, :, :]
        ) 

    return tmp_field
     


def laplacian2d(in_field, tmp_field):
    """Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    in_field  -- input field (nx x ny x nz with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    num_halo  -- number of halo points
    
    Keyword arguments:
    extend    -- extend computation into halo-zone by this number of points
    """
    
    I, J, K = in_field.shape

    #num_halo is set to 1, extent to 0
    
   
    for i in range(1, I - 1):
        for j in range(1, J - 1):
        
            tmp_field[i, j, :] = (
                -4.0 * in_field[i, j, :]
                + in_field[i - 1, j, :]
                + in_field[i + 1, j, :]
                + in_field[i, j - 1, :]
                + in_field[i, j + 1, :]
                )  

    return tmp_field
    


def laplacian3d(in_field, tmp_field):
    """Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    num_halo  -- number of halo points
    
    Keyword arguments:
    extend    -- extend computation into halo-zone by this number of points
    """

    I, J, K = in_field.shape 
    #num_halo is set to 1, extent to 0
    

    for i in range(1, I -1):
        for j in range(1, J - 1):
            for k in range(1, K -1):
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

def FMA(in_field, in_field2, in_field3, tmp_field):
    """
    Pointwise stencil to test for fused multiply-add. 
    
    Parameters
    ----------
    in_field1,2,3  : input field (nx x ny x nz).
    tmp_field : result (must be of same size as in_field).
    
    Returns
    -------
    tmp_field : fused multiply-add applied to in_field.
    
    """

    tmp_field = in_field + in_field2 * in_field3

    return tmp_field


def lapoflap1d(in_field, tmp_field,tmp2_field):
    """Compute Laplacian of the Laplacian in i-direction using 2nd-order centered differences with an explicit nested loop in numba.
    
    in_field  -- input field (nx x ny x nz with halo in x- and y-direction)
    tmp_field -- result (must be same size as in_field)
    
    """

    I, J, K = in_field.shape

    #ib = 1
    #ie = -1

    for i in range(1, I -1):
        tmp_field[i,:, :] = (
            -2.0 * in_field[i, :, :] + in_field[i - 1, :, :] + in_field[i + 1, :, :]
        ) 
    
    for i in range(2, I -2):
        tmp2_field[i,:, :] = (
            -2.0 * tmp_field[i, :, :] + tmp_field[i - 1, :, :] + tmp_field[i + 1, :, :]
        ) 
    
    
    return tmp2_field

def lapoflap2d(in_field, tmp_field,tmp2_field):
    """Compute Laplacian of the Laplacian in i and j-direction using 2nd-order centered differences with an explicit nested loop in numba.
    
    in_field  -- input field (nx x ny x nz with halo in x- and y-direction)
    tmp_field -- result (must be same size as in_field)
    
    """
    
    I, J, K = in_field.shape

    
    # ib = 1
    # ie = -1
    # jb = 1
    # je = -1
   
    for i in range(1, I - 1):
        for j in range(1, J - 1):
        
            tmp_field[i, j, :] = (
                -4.0 * in_field[i, j, :]
                + in_field[i - 1, j, :]
                + in_field[i + 1, j, :]
                + in_field[i, j - 1, :]
                + in_field[i, j + 1, :]
                )  

    for i in range(2, I-2):
        for j in range(2, J + -2):
        
            tmp2_field[i, j, :] = (
                -4.0 * tmp_field[i, j, :]
                + tmp_field[i - 1, j, :]
                + tmp_field[i + 1, j, :]
                + tmp_field[i, j - 1, :]
                + tmp_field[i, j + 1, :]
                )  
    return tmp2_field

def lapoflap3d(in_field, tmp_field,tmp2_field):
    """Compute Laplacian of the Laplacian in i,j,k-direction using 2nd-order centered differences with an explicit nested loop in numba.
    
    in_field  -- input field (nx x ny x nz with halo in x- and y-direction)
    tmp2_field -- result (must be same size as in_field)
    
    """
    
    I, J, K = in_field.shape 

    # ib = 1
    # ie = -1
    # jb = 1
    # je = -1
    # kb = 1
    # ke = -1


    for i in range(1, I - 1):
        for j in range(1, J - 1):
            for k in range(1, K - 1):
                tmp_field[i, j, k] = (
                        -6.0 * in_field[i, j, k]
                        + in_field[i - 1, j, k]
                        + in_field[i + 1, j, k]
                        + in_field[i, j - 1, k]
                        + in_field[i, j + 1, k]
                        + in_field[i, j, k - 1]
                        + in_field[i, j, k + 1]
                )
    
    for i in range(2, I -2):
        for j in range(2, J + -2):
            for k in range(2, K - 2):
                tmp2_field[i, j, k] = (
                        -6.0 * tmp_field[i, j, k]
                        + tmp_field[i - 1, j, k]
                        + tmp_field[i + 1, j, k]
                        + tmp_field[i, j - 1, k]
                        + tmp_field[i, j + 1, k]
                        + tmp_field[i, j, k - 1]
                        + tmp_field[i, j, k + 1]
                )

    return tmp2_field

    