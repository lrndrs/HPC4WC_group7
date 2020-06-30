# HPC4WC_group7
Comparison of high-level programming techniques for stencil computation (Coursework HPC4WC 2020)

Numba (http://numba.pydata.org/ 
https://nyu-cds.github.io/python-numba/
https://www.youtube.com/watch?v=x58W9A2lnQc 

### Setup environment:
- open the console
- which python --> prints the current environment
- source HPC4WC_venv/bin/activate --> loads the correct environment
- check again with which python, if it worked

### Run stencil_main.py
- move to its folder
- run with:
python3 stencil_main.py --dim 3 --nx 10 --ny 10 --nz 10 --num_iter 100 --stencil_type test
  