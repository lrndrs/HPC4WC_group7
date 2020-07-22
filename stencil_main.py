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


from functions import field_validation 
#     (
#     create_new_infield,
#     create_val_infield,
#     save_new_outfield,
#     validate_outfield,
# )
from functions import serialization
    #evaluate.add_data
    #evaluate.save_runtime_as_df
    
from functions import stencils_numpy
    #(
    #stencils_numpy.test,
    #stencils_numpy.laplacian1d,
    #stencils_numpy.laplacian2d,
    #stencils_numpy.laplacian3d,
    #stencils_numpy.FMA,
    #stencils_numpy.lapoflap1d,
    #stencils_numpy.lapoflap2d,
    #stencils_numpy.lapoflap3d,
#)
from functions import stencils_numbajit
#(
#    test_numba,
#    laplacian_numba,
#    FMA_numba,    
#) 
from functions import stencils_numbaloop
#(
#    laplacian1d,    
#    laplacian2d,
#    laplacian3d,
#) 
from functions import stencils_numbastencil 
#(
#    laplacian1d,
#    laplacian2d,
#    laplacian3d,
#    laplacian1d_numbastencil_help,
#    laplacian2d_numbastencil_help,
#    laplacian3d_numbastencil_help,
#)
from functions import stencils_numbavectorize
#(
#    FMA,
#)  # , laplacian1d_numbavectorize

from functions.halo_functions import update_halo, add_halo_points, remove_halo_points
#from numba import jit, njit
# from functions.gt4py_numpy import test_gt4py
# import gt4py
# import gt4py.gtscript as gtscript
# import gt4py.storage as gt_storage

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
    "--num_iter",
    type=int,
    default=1,
    help="Number of iterations",
)


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
    backend,
    stencil_name,
    num_iter=1,
    num_halo=2,
    plot_result=False,
    create_field=True,
    field_name="test",
    df_name="df",
    save_runtime=False
):
    """Driver for high-level comparison of stencil computation. HPC4WC group 7 coursework."""

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
        print(
            "please make sure you choose one of the following stencil: {}".format(
                stencil_name_list
            )
        )
        sys.exit(0)

    backend_list = ["numpy", "numbajit", "numbajit_inplace", "numbaloop", "numbastencil", "numbavectorize", "gt4py"]
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
    
    if create_field == False:
        in_field = field_validation.create_val_infield(nx, ny, nz,field_name)
    

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
    if backend == "numpy":#("numpy" or "numbajit"):
        if stencil_name == "test":
            stencils_numpy.test(in_field)

        if stencil_name == "laplacian1d":
            stencils_numpy.laplacian1d(in_field, tmp_field, num_halo=num_halo, extend=0)

        if stencil_name == "laplacian2d":
            stencils_numpy.laplacian2d(in_field, tmp_field, num_halo=num_halo, extend=0)

        if stencil_name == "laplacian3d":
            stencils_numpy.laplacian3d(in_field, tmp_field, num_halo=num_halo, extend=0)
            
        if stencil_name == "FMA":
            stencils_numpy.FMA(in_field, in_field2, in_field3, tmp_field, num_halo=num_halo, extend=0)
        
        if stencil_name == "lapoflap1d":
            stencils_numpy.lapoflap1d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)

        if stencil_name == "lapoflap2d":
            stencils_numpy.lapoflap2d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)

        if stencil_name == "lapoflap3d":
            stencils_numpy.lapoflap3d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)
    
    if backend == "numbajit":
        #if stencil_name == "laplacian1d":
        #    stencil = njit(stencils_numpy.laplacian1d, parallel=True, cache=True, fastmath=False)
        #    stencil(in_field, tmp_field, num_halo=num_halo, extend=0)
            
        if stencil_name == "test":
            stencils_numbajit.test(in_field)

        if stencil_name == "laplacian1d":
            stencils_numbajit.laplacian1d(in_field, tmp_field, num_halo=num_halo, extend=0)
            #print(stencils_numbajit.laplacian1d.inspect_types())

        if stencil_name == "laplacian2d":
            stencils_numbajit.laplacian2d(in_field, tmp_field, num_halo=num_halo, extend=0)

        if stencil_name == "laplacian3d":
            stencils_numbajit.laplacian3d(in_field, tmp_field, num_halo=num_halo, extend=0)
            
        if stencil_name == "FMA":
            stencils_numbajit.FMA(in_field, in_field2, in_field3, tmp_field, num_halo=num_halo, extend=0)
        
#         if stencil_name == "lapoflap1d":
#             stencils_numbajit.lapoflap1d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)

#         if stencil_name == "lapoflap2d":
#             stencils_numbajit.lapoflap2d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)

#         if stencil_name == "lapoflap3d":
#             stencils_numbajit.lapoflap3d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)   
    
    if backend == "numbajit_inplace":
        if stencil_name == "laplacian1d":
            stencil = njit(stencils_numpy.laplacian1d)#, parallel=False, cache=False, fastmath=False)
            stencil(in_field, tmp_field, num_halo=num_halo, extend=0)
        if stencil_name == "laplacian2d":
            stencil = njit(stencils_numpy.laplacian2d)
            stencil(in_field, tmp_field, num_halo=num_halo, extend=0)
        if stencil_name == "laplacian3d":
            stencil = njit(stencils_numpy.laplacian3d,)
            stencil(in_field, tmp_field, num_halo=num_halo, extend=0)
        if stencil_name == "FMA":
            stencil = njit(stencils_numpy.FMA)
            stencil(in_field, in_field2, in_field3, tmp_field, num_halo=num_halo, extend=0)    
        
    if backend == "numbaloop":
        if stencil_name == "laplacian1d":
            stencils_numbaloop.laplacian1d(
                in_field, tmp_field, num_halo=num_halo, extend=0
            )
        
        if stencil_name == "laplacian2d":
            stencils_numbaloop.laplacian2d(
                in_field, tmp_field, num_halo=num_halo, extend=0
            )

        if stencil_name == "laplacian3d":
            stencils_numbaloop.laplacian3d(
                in_field, tmp_field, num_halo=num_halo, extend=0
            )

    if backend == "numbastencil":

        if stencil_name == "laplacian1d":
            stencils_numbastencil.laplacian1d(in_field)

        if stencil_name == "laplacian2d":
            stencils_numbastencil.laplacian2d(in_field)

        if stencil_name == "laplacian3d":
            stencils_numbastencil.laplacian3d(in_field)
            
    if backend == "numbavectorize":

        if stencil_name == "FMA_numbavectorize":
            stencils_numbavectorize.FMA(in_field, in_field2, in_field3)

        # if stencil_name == "laplacian1d_numbavectorize":
        #    laplacian1d_numbavectorize( in_field)
        

    # time the actual work
    # Call the stencil chosen in stencil_name
    time_list = []
    for i in range(num_iter):
        if backend == "numpy":
            if stencil_name == "test":
                tic = time.time()
                out_field = stencils_numpy.test(in_field)
                toc = time.time()
            
            if stencil_name == "laplacian1d":
                tic = time.time()
                out_field = stencils_numpy.laplacian1d(in_field, tmp_field, num_halo=num_halo, extend=0)
                toc = time.time()

            if stencil_name == "laplacian2d":
                tic = time.time()
                out_field = stencils_numpy.laplacian2d(in_field, tmp_field, num_halo=num_halo, extend=0)
                toc = time.time()

            if stencil_name == "laplacian3d":
                tic = time.time()
                out_field = stencils_numpy.laplacian3d(in_field, tmp_field, num_halo=num_halo, extend=0)
                toc = time.time()
                
            if stencil_name == "FMA":
                tic = time.time()
                out_field = stencils_numpy.FMA(
                    in_field, in_field2, in_field3, tmp_field, num_halo=num_halo, extend=0)
                toc = time.time()
                
            if stencil_name == "lapoflap1d":
                tic = time.time()
                out_field = stencils_numpy.lapoflap1d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)
                toc = time.time()

            if stencil_name == "lapoflap2d":
                tic = time.time()
                out_field = stencils_numpy.lapoflap2d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)
                toc = time.time()

            if stencil_name == "lapoflap3d":
                tic = time.time()
                out_field = stencils_numpy.lapoflap3d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)
                toc = time.time()
                
        if backend == "numbajit":
            if stencil_name == "test":
                tic = time.time()
                out_field = stencils_numbajit.test(in_field)
                toc = time.time()
            
            if stencil_name == "laplacian1d":
                tic = time.time()
                out_field = stencils_numbajit.laplacian1d(in_field, tmp_field, num_halo=num_halo, extend=0)
                toc = time.time()

            if stencil_name == "laplacian2d":
                tic = time.time()
                out_field = stencils_numbajit.laplacian2d(in_field, tmp_field, num_halo=num_halo, extend=0)
                toc = time.time()

            if stencil_name == "laplacian3d":
                tic = time.time()
                out_field = stencils_numbajit.laplacian3d(in_field, tmp_field, num_halo=num_halo, extend=0)
                toc = time.time()
                
            if stencil_name == "FMA":
                tic = time.time()
                out_field = stencils_numbajit.FMA(
                    in_field, in_field2, in_field3, tmp_field, num_halo=num_halo, extend=0)
                toc = time.time()
                
#             if stencil_name == "lapoflap1d":
#                 tic = time.time()
#                 out_field = stencils_numbajit.lapoflap1d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)
#                 toc = time.time()

#             if stencil_name == "lapoflap2d":
#                 tic = time.time()
#                 out_field = stencils_numbajit.lapoflap2d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)
#                 toc = time.time()

#             if stencil_name == "lapoflap3d":
#                 tic = time.time()
#                 out_field = stencils_numbajit.lapoflap3d(in_field, tmp_field, tmp_field, num_halo=2, extend=1)
#                 toc = time.time()
        
        if backend == "numbajit_inplace":
            if stencil_name == "laplacian1d":
                tic = time.time()
                out_field = stencil(in_field, tmp_field, num_halo=num_halo, extend=0)
                toc = time.time()
            if stencil_name == "laplacian2d":
                tic = time.time()
                out_field = stencil(in_field, tmp_field, num_halo=num_halo, extend=0)
                toc = time.time()
            if stencil_name == "laplacian3d":
                tic = time.time()
                out_field = stencil(in_field, tmp_field, num_halo=num_halo, extend=0)
                toc = time.time()
            if stencil_name == "FMA":
                tic = time.time()
                out_field = stencil(in_field, in_field2, in_field3, tmp_field, num_halo=num_halo, extend=0) 
                toc = time.time()
            
        if backend == "numbaloop":
            if stencil_name == "laplacian1d":
                tic = time.time()
                out_field = stencils_numbaloop.laplacian1d(
                    in_field, tmp_field, num_halo=num_halo, extend=0
                )
                toc = time.time()

            if stencil_name == "laplacian2d":
                tic = time.time()
                out_field = stencils_numbaloop.laplacian2d(
                    in_field, tmp_field, num_halo=num_halo, extend=0
                )
                toc = time.time()

            if stencil_name == "laplacian3d":
                tic = time.time()
                out_field = stencils_numbaloop.laplacian3d(
                    in_field, tmp_field, num_halo=num_halo, extend=0
                )
                toc = time.time()

        if backend == "numbastencil":
            if stencil_name == "laplacian1d":
                tic = time.time()
                out_field = stencils_numbastencil.laplacian1d(in_field)
                toc = time.time()

            if stencil_name == "laplacian2d":
                tic = time.time()
                out_field = stencils_numbastencil.laplacian2d(in_field)
                toc = time.time()

            if stencil_name == "laplacian3d":
                tic = time.time()
                out_field = stencils_numbastencil.laplacian3d(in_field)
                toc = time.time()

        if backend == "numbavectorize":
            if stencil_name == "FMA":
                tic = time.time()
                out_field = stencils_numbavectorize.FMA(in_field, in_field2, in_field3)
                toc = time.time()
        # if stencil_name == "laplacian1d_numbavectorize":
        #    tic = time.time()
        #    laplacian1d_numbavectorize( in_field)
        #    toc = time.time()

        if backend == "gt4py":
            
            if stencil_name == "test_gt4py":
                origin = (2, 2, 0) # What does this do???
                dtype = np.float64
                backend_opt = "numpy"
                in_storage = gt_storage.from_array(
                    in_field, backend_opt, default_origin=origin, dtype=dtype 
                )
                out_storage = gt_storage.from_array(
                    tmp_field, backend_opt, default_origin=origin, dtype=dtype, 
                )
                coeff_storage = gt_storage.from_array(
                    in_field2, backend_opt, default_origin=origin, dtype=dtype,
                )
                tic = time.time()
                test_gt4py(in_storage, out_storage, coeff_storage)
                toc = time.time()
    
    
        time_list.append(toc - tic)
    
     
    time_avg = np.average(time_list)
    time_stdev = np.std(time_list)
    time_total = sum(time_list)

    
    
    print(
        "Total worktime: {} s. In {} iteration(s) the average lapsed time for one run is {} +/- {} s".format(
            time_total, num_iter, time_avg,time_stdev
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
    else :
        time_avg_first_10 = np.nan
        time_avg_last_10 = np.nan
        

    # delete halo from out_field
    out_field = remove_halo_points(out_field, num_halo)

    # Save or validate Outfield
    if create_field == True:
        field_validation.save_new_outfield(out_field,field_name)
        valid_var = "-"

    if create_field == False:
        valid_var = field_validation.validate_outfield(out_field,field_name)
        # TODO: Save Elapsed Work Time in table for validation mode
    
    # Save individual runtimes
    if save_runtime:
        serialization.save_runtime_as_df(time_list)
        print('Runtime development saved in dataframe.')
        
    # Append row with calculated work to df 
    serialization.add_data(df_name, stencil_name, backend, nx, ny, nz, valid_var, field_name, num_iter, time_total, time_avg, time_stdev, time_avg_first_10, time_avg_last_10)

    if plot_result:
        plt.imshow(out_field[out_field.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("out_field.png")
        plt.close()
        # TODO: print in and out field as pdf plot


if __name__ == "__main__":
    main()
