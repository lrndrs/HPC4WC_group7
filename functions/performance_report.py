# ******************************************************
# Functions for Performance Report generation
# ******************************************************

import csv

def new_reportfile(report_name):
    """
    Generates a new CSV file and sets the header row of the performance report

    Parameters
    ----------
    report_name : Name of File on disk

    Returns
    -------
    None.

    """
    with open(report_name,'w') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow(['stencil type','dim','nx','ny','nz','Elapsed Time','Validation'])
    
    print('New report {} generated.'.format(report_name))
        

def append_row(report_name,stencil_type,dim_stencil,nx,ny,nz,elapsedtime,valid_var):
    """
    Appends a row with several variables to the CSV performance report.

    Parameters
    ----------
    report_name : Name of File on disk
    stencil_type : Stencil Type from stencil list
    dim_stencil : stencil dimension
    nx : field size in x-Direction.
    ny : field size in y-Direction.
    nz : field size in z-Direction.
    elapsedtime : measured work time
    valid_var : Boolean if Validation was successful

    Returns
    -------
    None.

    """
    with open(report_name, 'a') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([stencil_type,dim_stencil,nx,ny,nz,elapsedtime,valid_var])
    