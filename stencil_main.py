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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from functions.field_validation import (
    create_new_infield,
    create_val_infield,
    save_newoutfield,
    validate_outfield,
)
from functions import evaluate

from functions import stencils_numpy
from functions import stencils_numbaloop
from functions import stencils_numbastencil
from functions import stencils_gt4py

from functions.halo_functions import update_halo, add_halo_points, remove_halo_points
from numba import njit

import gt4py


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
    help='Options are ["numpy", "numbajit", "numbajit_inplace", numbaloop", "numbastencil", "numbavectorize", "gt4py"]',
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
@click.option(
    "--num_halo",
    type=int,
    default=2,
    help="Number of halo-pointers in x- and y-direction",
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
@click.option("--num_iter", type=int, default=1, help="Number of iterations")
@click.option(
    "--field_name",
    type=str,
    default="test",
    help="Name of the testfield, that will be created or from which will be validated. File ending is added automatically.",
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
def main(
    nx,
    ny,
    nz,
    stencil_name,
    backend,
    numba_parallel=False,
    gt4py_backend="numpy",
    num_iter=1,
    num_halo=2,
    plot_result=False,
    create_field=True,
    field_name="test",
    df_name="df",
    save_runtime=False,
):
    """Driver for high-level comparison of stencil computation."""

    assert 0 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 0 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 0 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert (
        0 < num_iter <= 1024 * 1024
    ), "You have to specify a reasonable value for num_iter"
    assert (
        0 < num_halo <= 256
    ), "Your have to specify a reasonable number of halo points"
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
        raise ValueError(
            "please make sure you choose one of the following stencil: {}".format(
                stencil_name_list
            )
        )

    backend_list = ["numpy", "numba_vector", "numba_loop", "numba_stencil", "gt4py"]
    if backend not in backend_list:
        raise ValueError(
            "please make sure you choose one of the following backends: {}".format(
                backend_list
            )
        )

    # create field for validation
    if create_field:
        in_field = create_new_infield(nx, ny, nz, field_name)
    else:
        in_field = create_val_infield(nx, ny, nz, field_name)

    # np.save('in_field', in_field)
    if plot_result:
        plt.ioff()
        plt.imshow(in_field[in_field.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("in_field.png")
        plt.close()

    # expand in_field to contain halo points
    in_field = add_halo_points(in_field, num_halo)
    in_field = update_halo(in_field, num_halo)

    # create additional fields
    in_field2 = np.ones_like(in_field)
    in_field3 = np.ones_like(in_field) * 4.2
    tmp_field = np.empty_like(in_field)
    out_field = np.empty_like(in_field)

    # convert numpy arrays into gt4py storages
    if backend == "gt4py":
        in_field = gt4py.storage.from_array(
            in_field, gt4py_backend, default_origin=(num_halo, num_halo, num_halo)
        )
        in_field2 = gt4py.storage.from_array(
            in_field2, gt4py_backend, default_origin=(num_halo, num_halo, num_halo)
        )
        in_field3 = gt4py.storage.from_array(
            in_field3, gt4py_backend, default_origin=(num_halo, num_halo, num_halo)
        )
        tmp_field = gt4py.storage.from_array(
            tmp_field, gt4py_backend, default_origin=(num_halo, num_halo, num_halo)
        )
        out_field = gt4py.storage.from_array(
            out_field, gt4py_backend, default_origin=(num_halo, num_halo, num_halo)
        )

    # import and possibly compile proper stencil object
    if backend == "numpy":
        stencil = eval(f"stencils_numpy.{stencil_name}")
    elif backend == "numba_vector":
        stencil = eval(f"stencils_numpy.{stencil_name}")
        stencil = njit(stencil, parallel=numba_parallel)
    elif backend == "numba_loop":
        stencil = eval(f"stencils_numbaloop.{stencil_name}")
        stencil = njit(stencil, parallel=numba_parallel)
    elif backend == "numba_stencil":
        stencil = eval(f"stencils_numbastencil.{stencil_name}")
        stencil = njit(stencil, parallel=numba_parallel)
    else:  # gt4py
        stencil = eval(f"stencils_gt4py.{stencil_name}")
        stencil = gt4py.gtscript.stencil(gt4py_backend, stencil)

    # warmup caches
    if backend in ("numpy", "numba_vector", "numba_loop"):
        if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
            stencil(in_field, tmp_field, num_halo=num_halo, extend=0)
        elif stencil_name == "FMA":
            stencil(
                in_field, in_field2, in_field3, tmp_field, num_halo=num_halo, extend=0
            )
        elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
            stencil(in_field, tmp_field, out_field, num_halo=2, extend=1)
    elif backend == "numba_stencil":
        if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
            stencil(in_field, tmp_field)
        elif stencil_name == "FMA":
            stencil(in_field, in_field2, in_field3, tmp_field)
        elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
            stencil(in_field, tmp_field, out_field)
    else:  # gt4py
        if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
            stencil(
                in_field,
                tmp_field,
                origin=(num_halo, num_halo, num_halo),
                domain=(nx, ny, nz),
            )
        elif stencil_name == "FMA":
            stencil(
                in_field,
                in_field2,
                in_field3,
                tmp_field,
                origin=(num_halo, num_halo, num_halo),
                domain=(nx, ny, nz),
            )
        elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
            stencil(
                in_field,
                tmp_field,
                out_field,
                origin=(num_halo, num_halo, num_halo),
                domain=(nx, ny, nz),
            )

    # time the actual work
    # Call the stencil chosen in stencil_name
    time_list = []
    for i in range(num_iter):
        if backend in ("numpy", "numba_vector", "numba_loop"):
            if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
                tic = time.time()
                stencil(in_field, tmp_field, num_halo=num_halo, extend=0)
                toc = time.time()
            elif stencil_name == "FMA":
                tic = time.time()
                stencil(
                    in_field,
                    in_field2,
                    in_field3,
                    tmp_field,
                    num_halo=num_halo,
                    extend=0,
                )
                toc = time.time()
            elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
                tic = time.time()
                stencil(in_field, tmp_field, out_field, num_halo=2, extend=1)
                toc = time.time()
        elif backend == "numba_stencil":
            if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
                tic = time.time()
                stencil(in_field, tmp_field)
                toc = time.time()
            elif stencil_name == "FMA":
                tic = time.time()
                stencil(in_field, in_field2, in_field3, tmp_field)
                toc = time.time()
            elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
                tic = time.time()
                stencil(in_field, tmp_field, out_field)
                toc = time.time()
        else:  # gt4py
            if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
                tic = time.time()
                stencil(
                    in_field,
                    tmp_field,
                    origin=(num_halo, num_halo, num_halo),
                    domain=(nx, ny, nz),
                )
                toc = time.time()
            elif stencil_name == "FMA":
                tic = time.time()
                stencil(
                    in_field,
                    in_field2,
                    in_field3,
                    tmp_field,
                    origin=(num_halo, num_halo, num_halo),
                    domain=(nx, ny, nz),
                )
                toc = time.time()
            elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
                tic = time.time()
                stencil(
                    in_field,
                    tmp_field,
                    out_field,
                    origin=(num_halo, num_halo, num_halo),
                    domain=(nx, ny, nz),
                )
                toc = time.time()

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

    # delete halo from out_field
    out_field = remove_halo_points(out_field, num_halo)

    # Save or validate Outfield
    if create_field == True:
        save_newoutfield(out_field, field_name)
        valid_var = "-"

    if create_field == False:
        valid_var = validate_outfield(out_field, field_name)
        # TODO: Save Elapsed Work Time in table for validation mode

    # Save individual runtimes
    if save_runtime:
        evaluate.runtimedevelopment(time_list)
        print("Runtime development saved in dataframe.")

    # Append row with calculated work to df
    evaluate.add_data(
        df_name,
        stencil_name,
        backend,
        nx,
        ny,
        nz,
        valid_var,
        field_name,
        num_iter,
        time_total,
        time_avg,
        time_stdev,
        time_avg_first_10,
        time_avg_last_10,
    )

    if plot_result:
        plt.imshow(out_field[out_field.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("out_field.png")
        plt.close()
        # TODO: print in and out field as pdf plot


if __name__ == "__main__":
    main()
