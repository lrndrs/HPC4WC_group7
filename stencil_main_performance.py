# ******************************************************
#     Program: stencil_main.py
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
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage


# matplotlib.use("Agg")
# import matplotlib.pyplot as plt


# from functions import field_validation
# #     (
# #     create_new_infield,
# #     create_val_infield,
# #     save_new_outfield,
# #     validate_outfield,
# # )
from functions import serialization

from functions import stencils_numpy
from functions import stencils_numba_vector_decorator
from functions import stencils_numba_loop
from functions import stencils_numba_stencil
from functions import stencils_gt4py

from functions.halo_functions import update_halo, add_halo_points, remove_halo_points


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
    required=True,  # changed here
    help='Specify which stencil to use. Options are ["test", "laplacian1d", "laplacian2d","laplacian3d","FMA","lapoflap1d", "lapoflap2d", "lapoflap3d","test_gt4py"]',
)
@click.option(
    "--backend",
    type=str,
    required=True,
    help='Options are ["numpy", "numba_vector_function", "numba_vector_decorator", numba_loop", "numba_stencil", "numbavectorize", "gt4py"]',
)
@click.option(
    "--num_iter", type=int, default=1, help="Number of iterations",
)
@click.option(
    "--df_name",
    type=str,
    default="df",
    help="Name of evaluation dataframe. A new name creates a new df, the same name adds a column to the already existing df.",
)
@click.option(
    "--save_runtime",
    type=bool,
    default=False,
    help="Save the individual runtimes into a df.",
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
    num_iter=1,
    df_name="df",
    save_runtime=False,
    numba_parallel=False,
    gt4py_backend="numpy",
):
    """Performance assesment driver for high-level comparison of stencil computation in python."""

    assert 0 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 0 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 0 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert (
        0 < num_iter <= 1024 * 1024
    ), "You have to specify a reasonable value for num_iter"

    stencil_name_list = [  # changed here
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

    backend_list = [
        "numpy",
        "numba_vector_function",
        "numba_vector_decorator",
        "numba_loop",
        "numba_stencil",
        "gt4py",
    ]
    if backend not in backend_list:
        print(
            "please make sure you choose one of the following backends: {}".format(
                backend_list
            )
        )
        sys.exit(0)

    # TODO: Create check for gt4py_backend

    # alpha = 1.0 / 32.0
    # dim = 3

    # create field for validation
    # if create_field == True:
    #     in_field = field_validation.create_new_infield(nx, ny, nz,field_name)

    # else:
    #     in_field = field_validation.create_val_infield(nx, ny, nz,field_name)

    # # np.save('in_field', in_field)
    # if plot_result:
    #     plt.ioff()
    #     plt.imshow(in_field[in_field.shape[0] // 2, :, :], origin="lower")
    #     plt.colorbar()
    #     plt.savefig("in_field.png")
    #     plt.close()

    # Create random infield
    in_field = np.random.rand(nx, ny, nz)
    # expand in_field to contain halo points

    # define value of num_halo
    if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):  # changed here
        num_halo = 1
    elif stencil_name in (
        "lapoflap1d",
        "lapoflap2d",
        "lapoflap3d",
        "test_gt4py",
    ):  # changed here
        num_halo = 2
    else:  # FMA and test
        num_halo = 0

    in_field = add_halo_points(in_field, num_halo)
    in_field = update_halo(in_field, num_halo)

    # create additional fields
    in_field2 = np.ones_like(in_field)
    in_field3 = np.ones_like(in_field) * 4.2
    tmp_field = np.empty_like(in_field)
    out_field = np.empty_like(in_field)

    # create fields for gt4py #changed here
    if backend == "gt4py":
        if stencil_name in ["test_gt4py", "laplacian3d", "lapoflap3d"]:
            origin = (num_halo, num_halo, num_halo)
        elif stencil_name in ["laplacian1d", "lapoflap1d"]:
            origin = (num_halo, 0, 0)
        elif stencil_name in ["laplacian2d", "lapoflap2d"]:
            origin = (num_halo, num_halo, 0)
        elif stencil_name == "FMA":
            origin = (0, 0, 0)

        in_field = gt4py.storage.from_array(
            in_field, gt4py_backend, default_origin=origin
        )
        tmp_field = gt4py.storage.from_array(
            tmp_field, gt4py_backend, default_origin=origin
        )
        in_field2 = gt4py.storage.from_array(
            in_field2, gt4py_backend, default_origin=origin
        )
        in_field3 = gt4py.storage.from_array(  # changed here
            in_field3, gt4py_backend, default_origin=origin
        )
        out_field = gt4py.storage.from_array(
            out_field, gt4py_backend, default_origin=origin
        )

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
    else:  # gt4py
        stencil = eval(f"stencils_gt4py.{stencil_name}")
        stencil = gt4py.gtscript.stencil(gt4py_backend, stencil)

    # warm-up caches
    if backend in (
        "numpy",
        "numba_vector_function",
        "numba_vector_decorator",
        "numba_loop",
        "numba_stencil",
    ):  # changed
        if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
            stencil(in_field, out_field, num_halo=num_halo)  # changed
        elif stencil_name == "FMA":
            stencil(
                in_field, in_field2, in_field3, tmp_field, num_halo=num_halo
            )  # changed
        elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
            stencil(in_field, tmp_field, out_field, num_halo=2)  # changed
        else:  # Test
            stencil(in_field)

    #     elif backend in ("numba_loop","numba_stencil"):#changed
    #         if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
    #             stencil(in_field, tmp_field)
    #         elif stencil_name == "FMA":
    #             stencil(
    #                 in_field, in_field2, in_field3, tmp_field)
    #         elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
    #             stencil(in_field, tmp_field, out_field)
    #         else: #Test
    #             stencil(in_field)

    else:  # gt4py  #changed here
        if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d", "test_gt4py"):
            stencil(
                in_field, out_field, origin=origin, domain=(nx, ny, nz),
            )
        elif stencil_name == "FMA":
            stencil(
                in_field,
                in_field2,
                in_field3,
                out_field,
                origin=origin,
                domain=(nx, ny, nz),
            )
        elif stencil_name in ("lapoflap1", "lapoflap2d", "lapoflap3d"):
            stencil(
                in_field, tmp_field, out_field, origin=origin, domain=(nx, ny, nz),
            )
    #     #else: test

    # ----
    # time the actual work
    # Call the stencil chosen in stencil_name
    time_list = []
    for i in range(num_iter):

        if backend in (
            "numpy",
            "numba_vector_function",
            "numba_vector_decorator",
            "numba_loop",
            "numba_stencil",
        ):  # changed
            if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
                tic = time.time()
                out_field = stencil(in_field, tmp_field, num_halo=num_halo)  # changed
                toc = time.time()
            elif stencil_name == "FMA":
                tic = time.time()
                out_field = stencil(
                    in_field, in_field2, in_field3, tmp_field, num_halo=num_halo
                )  # changed
                toc = time.time()
            elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
                tic = time.time()
                out_field = stencil(
                    in_field, tmp_field, out_field, num_halo=2
                )  # changed
                toc = time.time()
            else:  # Test
                tic = time.time()
                out_field = stencil(in_field)
                toc = time.time()

        #         elif backend in ("numba_loop","numba_stencil"):#changed
        #             if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
        #                 tic = time.time()
        #                 out_field=stencil(in_field, tmp_field)
        #                 toc = time.time()
        #             elif stencil_name == "FMA":
        #                 tic = time.time()
        #                 out_field=stencil(
        #                     in_field, in_field2, in_field3, tmp_field)
        #                 toc = time.time()
        #             elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
        #                 tic = time.time()
        #                 out_field=stencil(in_field, tmp_field, out_field)
        #                 toc = time.time()
        #             else: #Test
        #                 tic = time.time()
        #                 out_field=stencil(in_field)
        #                 toc = time.time()

        else:  # gt4py  #changed here
            if stencil_name in (
                "laplacian1d",
                "laplacian2d",
                "laplacian3d",
                "test_gt4py",
            ):
                tic = time.time()
                stencil(
                    in_field, out_field, origin=origin, domain=(nx, ny, nz),
                )
                toc = time.time()
            elif stencil_name == "FMA":
                tic = time.time()
                stencil(
                    in_field,
                    in_field2,
                    in_field3,
                    out_field,
                    origin=origin,
                    domain=(nx, ny, nz),
                )
                toc = time.time()
            elif stencil_name in ("lapoflap1", "lapoflap2d", "lapoflap3d"):
                tic = time.time()
                stencil(
                    in_field, tmp_field, out_field, origin=origin, domain=(nx, ny, nz),
                )
                toc = time.time()
                # else: test
        time_list.append(toc - tic)

    time_avg = np.average(time_list)
    time_stdev = np.std(time_list)
    time_total = sum(time_list)

    print(
        "Total worktime: {} s. In {} iteration(s) the average lapsed time for one run is {} +/- {} s".format(
            time_total, num_iter, time_avg, time_stdev
        )
    )

    if num_iter >= 20:
        time_avg_first_10 = sum(time_list[0:10]) / 10
        time_avg_last_10 = sum(time_list[-11:-1]) / 10
        print(
            "The average elapsed time of the first 10 run is {} and of the last 10 values is {}".format(
                time_avg_first_10, time_avg_last_10
            )
        )
    else:
        time_avg_first_10 = np.nan
        time_avg_last_10 = np.nan

    # Save into df for further processing
    # Save runtimes
    if save_runtime:
        serialization.save_runtime_as_df(time_list)
        print("Individual runtime saved in dataframe.")

    # Append row with calculated work to df
    serialization.add_data(
        df_name,
        stencil_name,
        backend,
        numba_parallel,
        gt4py_backend,
        nx,
        ny,
        nz,
        num_iter,
        time_total,
        time_avg,
        time_stdev,
        time_avg_first_10,
        time_avg_last_10,
    )


if __name__ == "__main__":
    main()
