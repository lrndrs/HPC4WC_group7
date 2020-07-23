# ******************************************************
#     Program: stencil_main_validation.py
#      Author: HPC4WC Group 7
#        Date: 02.07.2020
# Description: Access different stencil functions via Commandline (click)
# ******************************************************

import time
import numpy as np
import click
import matplotlib
import sys
from numba import njit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from functions import field_validation 
from functions.halo_functions import update_halo, add_halo_points, remove_halo_points

from functions import stencils_numpy
from functions import stencils_numba_vector_decorator
from functions import stencils_numba_loop
from functions import stencils_numba_stencil 

# from functions.gt4py_numpy import test_gt4py
# import gt4py
# import gt4py.gtscript as gtscript
# import gt4py.storage as gt_storage



@click.command()
@click.option(
    "--nx", type=int, required=True, help="Number of gridpoints in x-direction"
)
@click.option(
    "--ny", type=int, required=True, help="Number of gridpoints in y-direction"
)
@click.option(
    "--nz", type=int, required=True, help="Number of gridpoints in z-direction"
)

@click.option(
    "--stencil_name",
    type=str,
    required=True,
    help='Specify which stencil to use. Options are ["test", "laplacian1d", "laplacian2d","laplacian3d","FMA","lapoflap1d", "lapoflap2d", "lapoflap3d", "test_gt4py"]',
)
@click.option(
    "--backend",
    type=str,
    required=True,
    help='Options are ["numpy", "numba_vector_function", "numba_vector_decorator", numba_loop", "numba_stencil", "numbavectorize", "gt4py"]',
)

@click.option(
    "--plot_result", type=bool, default=False, help="Make a plot of the result?"
)
@click.option(
    "--create_field",
    type=bool,
    default=True,
    help="Create a Field (True) or Validate from saved field (False)",
)

@click.option(
    "--field_name",
    type=str,
    default="test",
    help="Name of the testfield, that will be created or from which will be validated. File ending is added automatically.",
)

@click.option(
    "--numba_parallel",
    type=bool,
    default=False,
    help="True to enable parallel execution of Numba stencils.",
)
@click.option(
    "--gt4py_backend",
    type=str,
    default="numpy",
    help="GT4Py backend. Options are: numpy, gtx86, gtmc, gtcuda.",
)


def main(
    nx,
    ny,
    nz,
    backend,
    stencil_name,
    plot_result=False,
    create_field=True,
    field_name="test",
    numba_parallel=False,
    gt4py_backend="numpy"
):
    """Field validation driver for high-level comparison of stencil computation in python."""

    assert 0 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 0 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 0 < nz <= 1024, "You have to specify a reasonable value for nz"
    
    stencil_name_list = [
        "test",
        "laplacian1d",
        "laplacian2d",
        "laplacian3d",
        "FMA",
        "lapoflap1d",
        "lapoflap2d",
        "lapoflap3d",
        "test_gt4py",
    ]
    if stencil_name not in stencil_name_list:
        print(
            "please make sure you choose one of the following stencil: {}".format(
                stencil_name_list
            )
        )
        sys.exit(0)

    backend_list = ["numpy", "numba_vector_function", "numba_vector_decorator", "numba_loop", "numba_stencil", "gt4py"]
    if backend not in backend_list:
        print(
            "please make sure you choose one of the following backends: {}".format(
                backend_list
            )
        )
        sys.exit(0)
    #alpha = 1.0 / 32.0
    #dim = 3

    # create field for validation
    if create_field == True:
        in_field = field_validation.create_new_infield(nx, ny, nz,field_name)
    
    else:
        in_field = field_validation.create_val_infield(nx, ny, nz,field_name)
    
    # expand in_field to contain halo points
    #define value of num_halo
    if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
        num_halo = 1
    elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
        num_halo = 2
    else: #FMA and test
        num_halo = 0
        
    in_field = add_halo_points(in_field, num_halo)
    in_field = update_halo(in_field, num_halo)

    # plot result as png
    if plot_result:
        plt.ioff()
        plt.imshow(in_field[in_field.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("testfield/in_field.png")
        plt.close()

    # create additional fields
    in_field2 = np.ones_like(in_field)
    in_field3 = np.ones_like(in_field) * 4.2
    tmp_field = np.empty_like(in_field)
    out_field = np.empty_like(in_field)

#----

    # import and possibly compile proper stencil object
    if backend == "numpy":
        stencil = eval(f"stencils_numpy.{stencil_name}")
    elif backend == "numba_vector_decorator":
        stencil = eval(f"stencils_numba_vector_decorator.{stencil_name}")
    elif backend == "numba_vector_function":
        stencil = eval(f"stencils_numpy.{stencil_name}")
        stencil = njit(stencil, parallel=numba_parallel)
    elif backend == "numba_loop":
        stencil = eval(f"stencils_numba_loop.{stencil_name}")
        stencil = njit(stencil, parallel=numba_parallel)
    elif backend == "numba_stencil":
        stencil = eval(f"stencils_numba_stencil.{stencil_name}")
        stencil = njit(stencil, parallel=numba_parallel)
    # else:  # gt4py
    #     stencil = eval(f"stencils_gt4py.{stencil_name}")
    #     stencil = gt4py.gtscript.stencil(gt4py_backend, stencil)
    
    #warm-up caches (and run stencil computation one time)
    if backend in ("numpy", "numba_vector_function","numba_vector_decorator"):
        if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
            stencil(in_field, tmp_field, num_halo=num_halo, extend=0)
        elif stencil_name == "FMA":
            stencil(
                in_field, in_field2, in_field3, tmp_field, num_halo=num_halo, extend=0
            )
        elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
            stencil(in_field, tmp_field, out_field, num_halo=2, extend=1)
        else: #Test
            stencil(in_field)
    
    elif backend in ("numba_loop","numba_stencil"):
        if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
            stencil(in_field, tmp_field)
        elif stencil_name == "FMA":
            stencil(
                in_field, in_field2, in_field3, tmp_field)
        elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
            stencil(in_field, tmp_field, out_field)
        else: #Test
            stencil(in_field)
    
    # else:  # gt4py
    #     if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
    #         stencil(
    #             in_field,
    #             tmp_field,
    #             origin=(num_halo, num_halo, num_halo),
    #             domain=(nx, ny, nz),
    #         )
    #     elif stencil_name == "FMA":
    #         stencil(
    #             in_field,
    #             in_field2,
    #             in_field3,
    #             tmp_field,
    #             origin=(num_halo, num_halo, num_halo),
    #             domain=(nx, ny, nz),
    #         )
    #     elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
    #         stencil(
    #             in_field,
    #             tmp_field,
    #             out_field,
    #             origin=(num_halo, num_halo, num_halo),
    #             domain=(nx, ny, nz),
    #         )
    #     #else: test
        

    # delete halo from out_field
    out_field = remove_halo_points(out_field, num_halo)

    # Save or validate Outfield
    if create_field == True:
        field_validation.save_new_outfield(out_field, field_name)
        #valid_var = "-"

    elif create_field == False:
        field_validation.validate_outfield(out_field, field_name, stencil_name, backend)
    
    # Plot output field
    if plot_result:
        plt.imshow(out_field[out_field.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("testfields/out_field.png")
        plt.close()
        # TODO: print in and out field as pdf plot


if __name__ == "__main__":
    main()
