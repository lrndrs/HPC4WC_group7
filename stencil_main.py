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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from functions.fieldvalidation import create_new_infield, create_val_infield, save_newoutfield, validate_outfield
from functions.performancereport import new_reportfile, append_row
from functions.stencils import test, laplacian, FMA
#from functions.create_field import get_random_field
from functions.update_halo import update_halo
from functions.add_halo_points import add_halo_points
from functions.remove_halo_points import remove_halo_points



            
@click.command()
@click.option('--dim_stencil', type=int, required=True, help='Number of dimensions for stencil (1-3)')
@click.option('--nx', type=int, required=True, help='Number of gridpoints in x-direction')
@click.option('--ny', type=int, required=True, help='Number of gridpoints in y-direction')
@click.option('--nz', type=int, required=True, help='Number of gridpoints in z-direction')
@click.option('--num_iter', type=int, required=True, help='Number of iterations')
@click.option('--stencil_type', type=str, required=True, help='Specify which stencil to use. Options are [test, laplacian, FMA]')
@click.option('--num_halo', type=int, default=2, help='Number of halo-pointers in x- and y-direction')
@click.option('--plot_result', type=bool, default=False, help='Make a plot of the result?')
@click.option('--create_field', type=bool, default=True, help='Create a Field (True) or Validate from saved field (False)')
@click.option('--create_newreport', type=bool, default=True, help='Create a new report if a new field is generated (True/False)')
@click.option('--report_name', type=str, default='performance_report.csv', help='Specify a name for the csv performance report')

def main(dim_stencil, nx, ny, nz, num_iter, stencil_type, num_halo=2, plot_result=False, create_field=True,create_newreport=True,report_name='performance_report.csv'):
    """Driver for apply_diffusion that sets up fields and does timings"""
    
    assert 0 < nx <= 1024*1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024*1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024, 'You have to specify a reasonable value for nz'
    assert 0 < num_iter <= 1024*1024, 'You have to specify a reasonable value for num_iter'
    assert 0 < num_halo <= 256, 'Your have to specify a reasonable number of halo points'
    assert 0 <= dim_stencil <= 3, "Please choose between 0 and 3 dimensions"
    stencil_type_list = ["test", "laplacian", "FMA"]
    if stencil_type not in stencil_type_list:
        print("please make sure you choose one of the following stencil: {}".format(stencil_type_list))
        sys.exit(0)
    alpha = 1./32.
    dim = 3
    # TODO: create a field to validate results. 
    #create field
    if create_field==True:
        in_field = create_new_infield(nx,ny,nz)
        if create_newreport:
            new_reportfile(report_name)
        
    if create_field==False:
        in_field = create_val_infield(nx,ny,nz)
        
    
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
        out_field = FMA( in_field, dim_stencil=0, num_halo=num_halo, extend=0 )
        toc = time.time() 
        
    if stencil_type == "test":
        tic = time.time()
        out_field = test(in_field)
        toc = time.time()        
    
    print("Elapsed time for work = {} s".format(toc-tic) )
    elapsedtime = toc-tic
    

    #delete halo from out_field
    out_field = remove_halo_points(dim, out_field, num_halo)
    
    #Save or validate Outfield
    if create_field==True:
        save_newoutfield(out_field)
        valid_var = '-'
        
    if create_field==False:
        valid_var = validate_outfield(out_field)
        #TODO: Save Elapsed Work Time in table for validation mode
    
    # Append row with calculated work to report
    append_row(report_name,stencil_type,nx,ny,nz,elapsedtime,valid_var)
    
    if plot_result:
        plt.imshow(out_field[out_field.shape[0] // 2, :, :], origin='lower')
        plt.colorbar()
        plt.savefig('out_field.png')
        plt.close()
        #TODO: print in and out field as pdf plot
        
    


if __name__ == '__main__':
    main()
    


