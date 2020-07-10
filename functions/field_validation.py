# ******************************************************
# Functions for Field validation
# ******************************************************

import numpy as np


def create_new_infield(nx,ny,nz):
    """
    Creates a new 3D infield that is saved as .npy file and can be used for validation purposes.

    Parameters
    ----------
    nx : field size in x-Direction.
    ny : field size in y-Direction.
    nz : field size in z-Direction.

    Returns
    -------
    testfield : New In_Field used for stencil computation

    """
    testfield = np.random.rand(nx,ny,nz)
    np.save('test_infield.npy',testfield)
    
    return testfield

def create_val_infield(nx,ny,nz):
    """
    Loads an 3D infield that is saved as .npy file and can be used for validation purposes.
    Controls if new fieldsize is equivalent to the original field size.

    Parameters
    ----------
    nx : field size in x-Direction.
    ny : field size in y-Direction.
    nz : field size in z-Direction.

    Returns
    -------
    testfield : Field used for stencil computation

    """
    
    testfield = np.load('test_infield.npy')
    if (testfield.shape[0]!=nx) or (testfield.shape[1]!=ny) or (testfield.shape[0]!=nx):
            print('ERROR: New Infield has a different shape than the validation field.')
            exit()
    
    return testfield

def save_newoutfield(out_field):
    """
    Saves a new Out field to a .npy file.

    Parameters
    ----------
    out_field : 3D field after stencil computation

    Returns
    -------
    Print and save to .npy file

    """
    np.save('test_outfield.npy',out_field)
    print('New output field saved.')
    
    
def validate_outfield(out_field):
    """
    Reads in the original file and compares it to the current out-field. Validates the results of the stencil computation

    Parameters
    ----------
    out_field : 3D field after stencil computation

    Returns
    -------
    valid_var : boolean variable if Validation of array is true/false

    """
    testfield = np.load('test_outfield.npy')
    
    valid_var= np.all(np.equal(testfield,out_field))
    print('Result of field validation is:', valid_var)
    
    return valid_var

    

   