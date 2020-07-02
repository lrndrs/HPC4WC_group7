# ******************************************************
#     Program: stencil2d
#      Author: Oliver Fuhrer
#       Email: oliverf@vulcan.com
#        Date: 20.05.2020
# Description: Simple stencil example
# ******************************************************

import time
import numpy as np
import click
import matplotlib
import sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functions.stencils import test, laplacian, FMA
from functions.create_field import get_random_field
from functions.update_halo import update_halo
from functions.add_halo_points import add_halo_points



            
@click.command()
@click.option('--dim_stencil', type=int, required=True, help='Number of dimensions for stencil (1-3)')
@click.option('--nx', type=int, required=True, help='Number of gridpoints in x-direction')
@click.option('--ny', type=int, required=True, help='Number of gridpoints in y-direction')
@click.option('--nz', type=int, required=True, help='Number of gridpoints in z-direction')
@click.option('--num_iter', type=int, required=True, help='Number of iterations')
@click.option('--stencil_type', type=str, required=True, help='Specify which stencil to use. Options are [test, laplacian, FMA]')
@click.option('--num_halo', type=int, default=2, help='Number of halo-pointers in x- and y-direction')
@click.option('--plot_result', type=bool, default=False, help='Make a plot of the result?')

def main(dim_stencil, nx, ny, nz, num_iter, stencil_type, num_halo=2, plot_result=False):
    """Driver for apply_diffusion that sets up fields and does timings"""
    
    assert 0 < nx <= 1024*1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024*1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024, 'You have to specify a reasonable value for nz'
    assert 0 < num_iter <= 1024*1024, 'You have to specify a reasonable value for num_iter'
    assert 0 < num_halo <= 256, 'Your have to specify a reasonable number of halo points'
    assert 0 < dim_stencil <= 3, "Please choose between 1 and 3 dimensions"
    stencil_type_list = ["test", "laplacian", "FMA"]
    if stencil_type not in stencil_type_list:
        print("please make sure you choose one of the following stencil: {}".format(stencil_type_list))
        sys.exit(0)
    alpha = 1./32.
    dim = 3
    # TODO: create a field to validate results. 
    #create field
    in_field = get_random_field(dim, nx, ny, nz)
    
    #np.save('in_field', in_field)
    if plot_result:
        plt.ioff()
        plt.imshow(in_field[in_field.shape[0] // 2, :, :], origin='lower')
        plt.colorbar()
        plt.savefig('in_field.png')
        plt.close()
        
    # expand in_field to contain halo points
    in_field = add_halo_points(dim, in_field, num_halo)
    in_field = update_halo(dim, in_field, num_halo)
    
    #create additional fields
    tmp_field = np.empty_like(in_field)
    
    # warmup caches
    if stencil_type == "laplacian":
        laplacian( in_field, tmp_field, dim_stencil, num_halo=num_halo, extend=0 )
        
    if stencil_type == "test":
        test(in_field)
        
        
        
    # time the actual work
    # Call the stencil chosen in stencil_type
    if stencil_type == "laplacian":
        tic = time.time()
        out_field = laplacian( in_field, tmp_field, dim_stencil, num_halo=num_halo, extend=0 )
        toc = time.time() 
        
    if stencil_type == "FMA":
        tic = time.time()
        out_field = FMA( in_field, dim_stencil, num_halo=num_halo, extend=0 )
        toc = time.time() 
        
    if stencil_type == "test":
        tic = time.time()
        out_field = test(in_field)
        toc = time.time()        
    
    print("Elapsed time for work = {} s".format(toc-tic) )

    # TODO delet halo from out_field
    
    # np.save('out_field', out_field)
    if plot_result:
        plt.imshow(out_field[out_field.shape[0] // 2, :, :], origin='lower')
        plt.colorbar()
        plt.savefig('out_field.png')
        plt.close()


if __name__ == '__main__':
    main()
    


