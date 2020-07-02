#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:32:04 2020

@author: lau
"""

import numpy as np


def create_new_infield(nx,ny,nz):
    
    testfield = np.random.rand(nx,ny,nz)
    np.save('test_infield.npy',testfield)
    
    return testfield

def create_val_infield(nx,ny,nz):
    
    testfield = np.load('test_infield.npy')
    if (testfield.shape[0]!=nx) or (testfield.shape[1]!=ny) or (testfield.shape[0]!=nx):
            print('ERROR: New Infield has a different shape than the validation field.')
            exit()
    
    return testfield

def save_newoutfield(out_field):
    np.save('test_outfield.npy',out_field)
    print('New output field saved.')
    
    
def validate_outfield(out_field):
    testfield = np.load('test_outfield.npy')
    
    fprint= np.all(np.equal(testfield,out_field))
    print('Result of field validation is:', fprint)

    
    