import numpy as np
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
backend = "numpy"
dtype = np.float64

#def gt4p_stencil(in_field):

def test_gt4py(
    in_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):

        out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
            in_field[-1, -1, -1] - in_field[1, 1, 1] + in_field[0, 0, 0]
        )
        

def laplacian1d_gt4py(
    in_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):

        out_field[0,0,0] = (
            -2.0 * in_field[0,0,0] + in_field[-1, 0, 0] + in_field[1, 0, 0]
        ) 

def laplacian2d_gt4py(
    in_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):

        out_field[0,0,0] = (
            -4.0 * in_field[0,0,0] 
            + in_field[-1, 0, 0] 
            + in_field[1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, 1, 0]
        ) 

def laplacian3d_gt4py(
    in_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):

        out_field[0,0,0] = (
            -6.0 * in_field[0,0,0] 
            + in_field[-1, 0, 0] 
            + in_field[1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, 1, 0]
            + in_field[0, 0, -1]
            + in_field[0, 0, 1]
        ) 

def FMA_gt4py(
    in_field: gtscript.Field[dtype],
    in_field2: gtscript.Field[dtype],
    in_field3: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):

        out_field[0,0,0] = in_field[0,0,0] + in_field2[0,0,0] * in_field3[0,0,0]

def lapoflap1d_gt4py(
    in_field: gtscript.Field[dtype],
    tmp_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):
        tmp_field[0,0,0] = (
            -2.0 * in_field[0,0,0] + in_field[-1,0,0] + infield[1,0,0]
        )
        
        out_field[0,0,0] = (
            -2.0 * tmp_field[0,0,0] + in_field_[-1, 0, 0] + in_field[1,0,0])
    
        
def lapoflap2d_gt4py(
    in_field: gtscript.Field[dtype],
    tmp_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):

        tmp_field[0,0,0] = (
            -4.0 * in_field[0,0,0] 
            + in_field[-1, 0, 0] 
            + in_field[1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, 1, 0]
        ) 
        
        tout_field[0,0,0] = (
            -4.0 * tmp_field[0,0,0] 
            + tmp_field[-1, 0, 0] 
            + tmp_field[1, 0, 0]
            + tmp_field[0, -1, 0]
            + tmp_field[0, 1, 0]
        ) 
        
def lapoflap3d_gt4py(
    in_field: gtscript.Field[dtype],
    tmp_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):

        tmp_field[0,0,0] = (
            -6.0 * in_field[0,0,0] 
            + in_field[-1, 0, 0] 
            + in_field[1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, 1, 0]
            + in_field[0, 0, -1]
            + in_field[0, 0, 1]
        )
        
        out_field[0,0,0] = (
            -6.0 * tmp_field[0,0,0] 
            + tmp_field[-1, 0, 0] 
            + tmp_field[1, 0, 0]
            + tmp_field[0, -1, 0]
            + tmp_field[0, 1, 0]
            + tmp_field[0, 0, -1]
            + tmp_field[0, 0, 1]
        ) 
