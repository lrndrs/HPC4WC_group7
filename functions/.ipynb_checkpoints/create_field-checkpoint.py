import numpy as np


def get_random_field(dim, nx=1, ny=1, nz=1):
    #number of dimensions (1-3)
    #nx number of elements in x direction
    #ny number of elements in y direction
    #nz number of elements in z direction
    
    assert 0 < dim <= 3, 'You have to choose one of the following Dimension (dim) [1,2,3].'

    if dim == 1:
        print("dim = {}; nx = {}".format(dim, nx))
        return np.random.rand(nx)
    elif dim == 2:
        print("dim = {}; nx = {}; ny = {}".format(dim, nx, ny))
        return np.random.rand(nx,ny)
    elif dim == 3:
        print("dim = {}; nx = {}; ny = {}; nz = {}".format(dim, nx, ny, nz))
        return np.random.rand(nx,ny,nz)
    else:
        print("please give a number for dim between 1 and 3. You gave {}".format(dim))
        return