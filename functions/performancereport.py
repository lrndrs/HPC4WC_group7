#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:53:17 2020

@author: lau
"""

import csv

def new_reportfile(report_name):
    with open(report_name,'w') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow(['stencil type','nx','ny','nz','Elapsed Time','Validation'])
    
    print('New report {} generated.'.format(report_name))
        

def append_row(report_name,stencil_type,nx,ny,nz,elapsedtime,valid_var):
    with open(report_name, 'a') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([stencil_type,nx,ny,nz,elapsedtime,valid_var])
    