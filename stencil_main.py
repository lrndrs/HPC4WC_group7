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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from functions.field_validation import (
    create_new_infield,
    create_val_infield,
    save_newoutfield,
    validate_outfield,
)
from functions.performance_report import new_reportfile, append_row
from functions.stencils_numpy import (
    test,
    laplacian1d,
    laplacian2d,
    laplacian3d,
    FMA,
    lapoflap1d,
    lapoflap2d,
    lapoflap3d,
)
from functions.stencils_numba import (
    test_numba,
    laplacian_numba,
    laplacian_numbaloop,
    FMA_numba,
    laplacian1d_numbastencil,
    laplacian2d_numbastencil,
    laplacian3d_numbastencil,
    laplacian1d_numbastencil_help,
    laplacian2d_numbastencil_help,
    laplacian3d_numbastencil_help,
    FMA_numbavectorize,
)  # , laplacian1d_numbavectorize

from functions.halo_functions import update_halo, add_halo_points, remove_halo_points
from numba import jit

# from functions.create_field import get_random_field
# from functions.update_halo import update_halo
# from functions.add_halo_points import add_halo_points
# from functions.remove_halo_points import remove_halo_points


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
    "--num_iter", type=int, required=False, default=1, help="Number of iterations"
)
@click.option(
    "--stencil_name",
    type=str,
    required=True,
    help='Specify which stencil to use. Options are ["test", "laplacian1d", "laplacian2d","laplacian3d","FMA","test_numba","laplacian_numba","laplacian_numbaloop","FMA_numba", "laplacian1d_numbastencil","laplacian2d_numbastencil", "laplacian3d_numbastencil", "FMA_numbavectorize", "lapoflap1d", "lapoflap2d", "lapoflap3d"]',
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
@click.option(
    "--create_newreport",
    type=bool,
    default=True,
    help="Create a new report if a new field is generated (True/False)",
)
@click.option(
    "--report_name",
    type=str,
    default="performance_report.csv",
    help="Specify a name for the csv performance report",
)
def main(
    nx,
    ny,
    nz,
    num_iter,
    stencil_name,
    num_halo=2,
    plot_result=False,
    create_field=True,
    create_newreport=True,
    report_name="performance_report.csv",
):
    """Driver for apply_diffusion that sets up fields and does timings"""

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
        "test_numba",
        "laplacian_numba",
        "laplacian_numbaloop",
        "FMA_numba",
        "laplacian1d_numbastencil",
        "laplacian2d_numbastencil",
        "laplacian3d_numbastencil",
        "FMA_numbavectorize",
        "lapoflap1d",
        "lapoflap2d",
        "lapoflap3d",
    ]
    if stencil_name not in stencil_name_list:
        print(
            "please make sure you choose one of the following stencil: {}".format(
                stencil_name_list
            )
        )
        sys.exit(0)
    alpha = 1.0 / 32.0
    dim = 3

    # Set field constraints according to dim_stencil
    # if dim_stencil ==1:
    #     if nx!=1 or ny!=1:
    #       print('WARNING: Dimension is set to 1D, only nz value is considered.')
    #     nx,ny=1

    # if dim_stencil ==2:
    #     if nx!=1:
    #       print('WARNING: Dimension is set to 2D, only nz and ny values are considered.')
    #     nx=1

    # create field for validation
    if create_field == True:
        in_field = create_new_infield(nx, ny, nz)
        if create_newreport:
            new_reportfile(report_name)

    if create_field == False:
        in_field = create_val_infield(nx, ny, nz)

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

    # warmup caches
    if stencil_name == "laplacian1d":
        laplacian1d(in_field, tmp_field, num_halo=num_halo, extend=0)

    if stencil_name == "laplacian2d":
        laplacian2d(in_field, tmp_field, num_halo=num_halo, extend=0)

    if stencil_name == "laplacian3d":
        laplacian3d(in_field, tmp_field, num_halo=num_halo, extend=0)

    if stencil_name == "laplacian_numba":
        laplacian_numba(in_field, tmp_field, dim, num_halo=num_halo, extend=0)

    if stencil_name == "laplacian_numbaloop":
        laplacian_numbaloop(
            in_field, tmp_field, dim, num_halo=num_halo, extend=0
        )

    if stencil_name == "test":
        test(in_field)

    if stencil_name == "test_numba":
        test_numba(in_field)

    if stencil_name == "FMA":
        FMA(in_field, in_field2, in_field3, tmp_field, num_halo=num_halo, extend=0)

    if stencil_name == "FMA_numba":
        FMA_numba(in_field, dim=0, num_halo=num_halo, extend=0)

    if stencil_name == "laplacian1d_numbastencil":
        laplacian1d_numbastencil(in_field)

    if stencil_name == "laplacian2d_numbastencil":
        laplacian2d_numbastencil(in_field)

    if stencil_name == "laplacian3d_numbastencil":
        laplacian3d_numbastencil(in_field)

    if stencil_name == "FMA_numbavectorize":
        FMA_numbavectorize(in_field, in_field2, in_field3)

    if stencil_name == "lapoflap1d":
        lapoflap1d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)

    if stencil_name == "lapoflap2d":
        lapoflap2d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)

    if stencil_name == "lapoflap3d":
        lapoflap3d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)

    # if stencil_name == "laplacian1d_numbavectorize":
    #    laplacian1d_numbavectorize( in_field)

    # time the actual work
    # Call the stencil chosen in stencil_name
    time_list = []
    for i in range(num_iter):
        if stencil_name == "laplacian1d":
            tic = time.time()
            out_field = laplacian1d(in_field, tmp_field, num_halo=num_halo, extend=0)
            toc = time.time()

        if stencil_name == "laplacian2d":
            tic = time.time()
            out_field = laplacian2d(in_field, tmp_field, num_halo=num_halo, extend=0)
            toc = time.time()

        if stencil_name == "laplacian3d":
            tic = time.time()
            out_field = laplacian3d(in_field, tmp_field, num_halo=num_halo, extend=0)
            toc = time.time()

        if stencil_name == "laplacian_numba":
            tic = time.time()
            out_field = laplacian_numba(
                in_field, tmp_field, dim, num_halo=num_halo, extend=0
            )
            toc = time.time()

        if stencil_name == "laplacian_numbaloop":
            tic = time.time()
            out_field = laplacian_numbaloop(
                in_field, tmp_field, dim, num_halo=num_halo, extend=0
            )
            toc = time.time()

        if stencil_name == "FMA":
            tic = time.time()
            out_field = FMA(
                in_field, in_field2, in_field3, tmp_field, num_halo=num_halo, extend=0
            )
            toc = time.time()

        if stencil_name == "FMA_numba":
            tic = time.time()
            out_field = FMA_numba(in_field, dim=0, num_halo=num_halo, extend=0)
            toc = time.time()

        if stencil_name == "test":
            tic = time.time()
            out_field = test(in_field)
            toc = time.time()

        if stencil_name == "test_numba":
            tic = time.time()
            out_field = test_numba(in_field)
            toc = time.time()

        if stencil_name == "laplacian1d_numbastencil":
            tic = time.time()
            out_field = laplacian1d_numbastencil(in_field)
            toc = time.time()

        if stencil_name == "laplacian2d_numbastencil":
            tic = time.time()
            out_field = laplacian2d_numbastencil(in_field)
            toc = time.time()

        if stencil_name == "laplacian3d_numbastencil":
            tic = time.time()
            out_field = laplacian3d_numbastencil(in_field)
            toc = time.time()

        if stencil_name == "FMA_numbavectorize":
            tic = time.time()
            out_field = FMA_numbavectorize(in_field, in_field2, in_field3)
            toc = time.time()

        if stencil_name == "lapoflap1d":
            tic = time.time()
            out_field = lapoflap1d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)
            toc = time.time()

        if stencil_name == "lapoflap2d":
            tic = time.time()
            out_field = lapoflap2d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)
            toc = time.time()

        if stencil_name == "lapoflap3d":
            tic = time.time()
            out_field = lapoflap3d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)
            toc = time.time()

        time_list.append(toc - tic)
        # if stencil_name == "laplacian1d_numbavectorize":
        #    tic = time.time()
        #    laplacian1d_numbavectorize( in_field)
        #    toc = time.time()

    time_avg = sum(time_list) / len(time_list)
    print(
        "In {} iterations the average lapsed time for work is {} s".format(
            num_iter, time_avg
        )
    )

    if num_iter >= 20:
        time_avg_first_10 = sum(time_list[0:10]) / len(time_list[0:10])
        time_avg_last_10 = sum(time_list[-11:-1]) / len(time_list[-11:-1])
        print(
            "The average elapes time of the first 10 values is {} and of the last 10 values us {}".format(
                time_avg_first_10, time_avg_last_10
            )
        )

    # delete halo from out_field
    out_field = remove_halo_points(out_field, num_halo)

    # Save or validate Outfield
    if create_field == True:
        save_newoutfield(out_field)
        valid_var = "-"

    if create_field == False:
        valid_var = validate_outfield(out_field)
        # TODO: Save Elapsed Work Time in table for validation mode

    # Append row with calculated work to report
    append_row(report_name, stencil_name, dim, nx, ny, nz, time_avg, valid_var)

    if plot_result:
        plt.imshow(out_field[out_field.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("out_field.png")
        plt.close()
        # TODO: print in and out field as pdf plot


if __name__ == "__main__":
    main()
