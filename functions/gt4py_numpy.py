import numpy as np
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

backend = "numpy"
dtype = np.float64

# def gt4p_stencil(in_field):


def test_gt4py(
    in_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
    coeff: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):
        lap_field = 4.0 * in_field[0, 0, 0] - (
            in_field[1, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[0, 1, 0]
            + in_field[0, -1, 0]
        )
        res = lap_field[1, 0, 0] - lap_field[0, 0, 0]
        flx_field = 0 if (res * (in_field[1, 0, 0] - in_field[0, 0, 0])) > 0 else res
        res = lap_field[0, 1, 0] - lap_field[0, 0, 0]
        fly_field = 0 if (res * (in_field[0, 1, 0] - in_field[0, 0, 0])) > 0 else res
        out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
            flx_field[0, 0, 0]
            - flx_field[-1, 0, 0]
            + fly_field[0, 0, 0]
            - fly_field[0, -1, 0]
        )
