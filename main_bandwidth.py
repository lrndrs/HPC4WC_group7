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
import math
from numba import njit, cuda
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

try: 
    import cupy as cp
except ImportError:
        cp=np

from functions import serialization

from functions.timing import get_time

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
    "--backend",
    type=str,
    required=True,
    help='Options are ["cupy", "numba_cuda", "gt4py"]',
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

def main(
    nx,
    ny,
    nz,
    backend,
    num_iter=1,
    df_name="df",
    save_runtime=False
):
    """Driver to measure the bandwidth used by different backends."""

    assert 1 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 1 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 1 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert (
        0 < num_iter <= 1024 * 1024
    ), "You have to specify a reasonable value for num_iter"

    backend_list = [
        "numba_cuda",
        "gt4py",
        "cupy",
    ]
    if backend not in backend_list:
        print(
            "please make sure you choose one of the following backends: {}".format(
                backend_list
            )
        )
        sys.exit(0)

    # Create random infield
    in_field = np.random.rand(nx, ny, nz)

        # ----
    # time the data transefer
    time_list = []
    for i in range(num_iter):
        # create threads for numba_cuda:
        if backend == "numba_cuda":
            threadsperblock = (8,8,8)

            blockspergrid_x = math.ceil(in_field.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(in_field.shape[1] / threadsperblock[1])
            blockspergrid_z = math.ceil(in_field.shape[2] / threadsperblock[2])
            blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

            #numba_cudadevice:
            tic = time.time()
            in_field_d = cuda.to_device(in_field)
            toc = time.time()  

        # create fields for cupy
        if backend == "cupy":
            tic = get_time()
            in_field = cp.array(in_field)
            toc = get_time()                             


        # create fields for gt4py 
        if backend == "gt4py":
            in_field_gt = gt4py.storage.from_array(
                in_field, backend="gtcuda", default_origin=(0,0,0))
            tic=time.time()
            in_field_gt.synchronize()
            toc=time.time()
        
        time_list.append(toc - tic)

    time_avg = np.average(time_list[:])
    time_stdev = np.std(time_list[:])
    time_total = sum(time_list[:])

    print(
        "Total worktime: {} s. In {} iteration(s) the average lapsed time for one run is {} +/- {} s".format(
            time_total, num_iter, time_avg, time_stdev
        )
    )
    
    #compute size of transferred data
    num_elements = nx * ny * nz
    number_of_bytes = 8 * num_elements
    number_of_gbytes = number_of_bytes / 1024**3
    print("data transferred = {} GB".format(number_of_gbytes))

    # memory bandwidth
    memory_bandwidth_in_gbs = number_of_gbytes/time_avg
    memory_bandwidth_stdev = number_of_gbytes/time_stdev
    print("memory bandwidth = {:8.5f} GB/s".format(memory_bandwidth_in_gbs))

    ##theoretical peak memory bandwidth
    #f_ddr=2133*10**6
    #channels=4
    #width=64/8
    peak_bandwidth_in_gbs = 68.3#f_ddr*channels*width*1.e-9
    print("peak memory bandwidth = {} GB/s".format(peak_bandwidth_in_gbs))

    # compute fraction of peak
    fraction_of_peak_bandwidth = memory_bandwidth_in_gbs/peak_bandwidth_in_gbs
    fraction_of_peak_bandwidth_stdev= memory_bandwidth_stdev/peak_bandwidth_in_gbs
    print("%peak = {:8.5f}%".format(fraction_of_peak_bandwidth))


    # Save into df for further processing
    # Save runtimes
    if (save_runtime==True):
        serialization.save_runtime_as_df(time_list[:])
        print("Individual runtime saved in dataframe.")

    # Append row with calculated work to df
    serialization.add_data_bandwidth(
        df_name,
        backend,
        nx,
        ny,
        nz,
        num_iter,
        time_total,
        time_avg,
        time_stdev,
        number_of_gbytes,
        memory_bandwidth_in_gbs,
        memory_bandwidth_stdev,
        peak_bandwidth_in_gbs,
        fraction_of_peak_bandwidth,
        fraction_of_peak_bandwidth_stdev
    )



if __name__ == "__main__":
    main()
