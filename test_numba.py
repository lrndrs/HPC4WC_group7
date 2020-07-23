from numba import njit, stencil
import numpy as np


@stencil(neighborhood=((-1, 1), (-1, 1)))
def one_help(in_field):
    return in_field[0, 0] + 1


def one(in_field, out_field):
    one_help(in_field, out=out_field)


if __name__ == "__main__":
    in_field = np.zeros((10, 10))
    out_field = np.zeros_like(in_field)

    one = njit(one)

    one(in_field, out_field)

    print(f"Expected result: {(in_field.shape[0] - 2) * (in_field.shape[1] - 2)}")
    print(f"Actual result: {int(out_field.sum())}")
