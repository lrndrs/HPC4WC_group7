import numpy as np
from numba import stencil

# @stencil
# def test(in_field):
#     """
#     Simple test function that returns a copy of the in_field.
    
#     Parameters
#     ----------
#     in_field  : input field (nx x ny x nz).
    
#     Returns
#     -------
#     out_field : a copy of the in_field.
    
#     """
#     return in_field[0,0,0]



@stencil(neighborhood=((-1, 1), (-1, 1), (-1, 1)))
def laplacian1d_help(in_field):
    """
    Numpy function that computes the Laplacian of the in_field in i-direction. This function is called inside the function laplacian1d.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    out_field: in_field with Laplacian computed in i-direction.
    """
    return -2.0 * in_field[0, 0, 0] + in_field[-1, 0, 0] + in_field[+1, 0, 0]


def laplacian1d(in_field, out_field, num_halo=1):
    """
    Function that boosts the function laplacian1d_numbastencil_help
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    
    """
    laplacian1d_help(in_field, out= out_field)


@stencil(neighborhood=((-1, 1), (-1, 1), (-1, 1)))
def laplacian2d_help(in_field):
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



def laplacian2d(in_field, out_field, num_halo=1):
    """
    Function that boosts the function laplacian2d_numbastencil_help
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    
    """
    laplacian2d_help(in_field, out= out_field)


@stencil(neighborhood=((-1, 1), (-1, 1), (-1, 1)))
def laplacian3d_help(in_field):
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


def laplacian3d(in_field, out_field, num_halo=1):
    """
    Function that boosts the function laplacian3d_numbastencil_help
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    
    """
    return laplacian3d_help(in_field, out= out_field)
