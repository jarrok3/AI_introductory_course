import numpy as np

def shift_r(x):
    return np.concatenate((np.zeros_like(x[..., :, -1:]), x[..., :, :-1]), axis=-1)
def shift_l(x):
    return np.concatenate((x[..., :, 1:], np.zeros_like(x[..., :, :1])), axis=-1)
def shift_t(x):
    return np.concatenate((np.zeros_like(x[..., -1:, :]), x[..., :-1, :]), axis=-2)
def shift_b(x):
    return np.concatenate((x[..., 1:, :], np.zeros_like(x[..., :1, :])), axis=-2)


def evaluate(x, size=20):
    # Check if individual genome size is 400
    assert np.shape(x)[-1] == size * size
    # Take the genomme and resize into table 20x20, make the grid 3D
    grid = np.asarray(np.reshape(x, (-1, size, size)),dtype=int)
    points = np.minimum(
    shift_r(grid) + shift_l(grid) + shift_t(grid) + shift_b(grid),
    1 - grid
    )
    # reshape back to original dimensions
    # sum along the first axis to get the total number of points for each individual
    return points.reshape(np.shape(x)).sum(-1)
