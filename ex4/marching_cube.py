from collections import defaultdict
import numpy as np

from matplotlib import pyplot as plt

from scipy.spatial import KDTree
from time import perf_counter
import k3d
from trimesh import Trimesh

LUT_CUBE_IDX_TO_TRIANGLES_EDGE_IDXS = [
    [],
    [(0, 8, 3)],
    [(0, 1, 9)],
    [(1, 8, 3), (9, 8, 1)],
    [(1, 2, 10)],
    [(0, 8, 3), (1, 2, 10)],
    [(10, 9, 0), (0, 2, 10)],
    [(2, 8, 3), (2, 10, 8), (10, 9, 8)],
    [(3, 11, 2)],
    [(2, 0, 8), (8, 11, 2)],
    [(1, 9, 0), (2, 3, 11)],
    [(1, 11, 2), (1, 9, 11), (9, 8, 11)],
    [(3, 11, 1), (1, 11, 10)],
    [(0, 10, 1), (0, 8, 10), (11, 10, 8)],
    [(3, 9, 0), (3, 11, 9), (11, 10, 9)],
    [(9, 8, 11), (10, 9, 11)],
    [(4, 7, 8)],
    [(4, 3, 0), (7, 3, 4)],
    [(0, 1, 9), (8, 4, 7)],
    [(4, 1, 9), (4, 7, 1), (7, 3, 1)],
    [(1, 2, 10), (8, 4, 7)],
    [(3, 4, 7), (3, 0, 4), (1, 2, 10)],
    [(9, 2, 10), (9, 0, 2), (8, 4, 7)],
    [(2, 10, 9), (2, 9, 7), (2, 7, 3), (7, 9, 4)],
    [(8, 4, 7), (3, 11, 2)],
    [(11, 4, 7), (11, 2, 4), (2, 0, 4)],
    [(9, 0, 1), (8, 4, 7), (2, 3, 11)],
    [(4, 7, 11), (9, 4, 11), (9, 11, 2), (2, 1, 9)],
    [(3, 11, 10), (10, 1, 3), (7, 8, 4)],
    [(11, 10, 1), (4, 11, 1), (7, 11, 4), (4, 1, 0)],
    [(4, 7, 8), (3, 11, 9), (9, 11, 10), (9, 0, 3)],
    [(4, 7, 11), (4, 11, 9), (9, 11, 10)],
    [(9, 5, 4)],
    [(9, 5, 4), (0, 8, 3)],
    [(0, 5, 4), (1, 5, 0)],
    [(8, 5, 4), (8, 3, 5), (3, 1, 5)],
    [(1, 2, 10), (9, 5, 4)],
    [(3, 0, 8), (1, 2, 10), (4, 9, 5)],
    [(5, 2, 10), (5, 4, 2), (4, 0, 2)],
    [(2, 10, 5), (3, 2, 5), (3, 5, 4), (3, 4, 8)],
    [(9, 5, 4), (2, 3, 11)],
    [(0, 11, 2), (0, 8, 11), (4, 9, 5)],
    [(0, 5, 4), (0, 1, 5), (2, 3, 11)],
    [(2, 1, 5), (2, 5, 8), (2, 8, 11), (4, 8, 5)],
    [(10, 3, 11), (10, 1, 3), (9, 5, 4)],
    [(4, 9, 5), (0, 8, 1), (8, 10, 1), (8, 11, 10)],
    [(5, 4, 0), (5, 0, 11), (5, 11, 10), (11, 0, 3)],
    [(5, 4, 8), (5, 8, 10), (10, 8, 11)],
    [(9, 7, 8), (5, 7, 9)],
    [(9, 3, 0), (9, 5, 3), (5, 7, 3)],
    [(0, 7, 8), (0, 1, 7), (1, 5, 7)],
    [(1, 5, 3), (3, 5, 7)],
    [(9, 7, 8), (9, 5, 7), (10, 1, 2)],
    [(10, 1, 2), (9, 5, 0), (5, 3, 0), (5, 7, 3)],
    [(8, 0, 2), (8, 2, 5), (8, 5, 7), (10, 5, 2)],
    [(2, 10, 5), (2, 5, 3), (3, 5, 7)],
    [(7, 9, 5), (7, 8, 9), (3, 11, 2)],
    [(9, 5, 7), (9, 7, 2), (9, 2, 0), (2, 7, 11)],
    [(2, 3, 11), (0, 1, 8), (1, 7, 8), (1, 5, 7)],
    [(11, 2, 1), (11, 1, 7), (7, 1, 5)],
    [(9, 5, 8), (8, 5, 7), (10, 1, 3), (10, 3, 11)],
    [(5, 7, 0), (5, 0, 9), (7, 11, 0), (1, 0, 10), (11, 10, 0)],
    [(11, 10, 0), (11, 0, 3), (10, 5, 0), (8, 0, 7), (5, 7, 0)],
    [(5, 11, 10), (11, 5, 7)],
    [(10, 6, 5)],
    [(0, 8, 3), (6, 5, 10)],
    [(9, 0, 1), (5, 10, 6)],
    [(1, 8, 3), (1, 9, 8), (5, 10, 6)],
    [(1, 6, 5), (2, 6, 1)],
    [(1, 6, 5), (1, 2, 6), (3, 0, 8)],
    [(9, 6, 5), (9, 0, 6), (0, 2, 6)],
    [(5, 9, 8), (5, 8, 2), (5, 2, 6), (3, 2, 8)],
    [(2, 3, 11), (10, 6, 5)],
    [(11, 0, 8), (11, 2, 0), (10, 6, 5)],
    [(0, 1, 9), (2, 3, 11), (5, 10, 6)],
    [(5, 10, 6), (1, 9, 2), (9, 11, 2), (9, 8, 11)],
    [(6, 3, 11), (6, 5, 3), (5, 1, 3)],
    [(0, 8, 11), (0, 11, 5), (0, 5, 1), (5, 11, 6)],
    [(3, 11, 6), (0, 3, 6), (0, 6, 5), (0, 5, 9)],
    [(6, 5, 9), (6, 9, 11), (11, 9, 8)],
    [(5, 10, 6), (4, 7, 8)],
    [(4, 3, 0), (4, 7, 3), (6, 5, 10)],
    [(1, 9, 0), (5, 10, 6), (8, 4, 7)],
    [(10, 6, 5), (1, 9, 7), (1, 7, 3), (7, 9, 4)],
    [(6, 1, 2), (6, 5, 1), (4, 7, 8)],
    [(1, 2, 5), (5, 2, 6), (3, 0, 4), (3, 4, 7)],
    [(8, 4, 7), (9, 0, 5), (0, 6, 5), (0, 2, 6)],
    [(7, 3, 9), (7, 9, 4), (3, 2, 9), (5, 9, 6), (2, 6, 9)],
    [(3, 11, 2), (7, 8, 4), (10, 6, 5)],
    [(5, 10, 6), (4, 7, 2), (4, 2, 0), (2, 7, 11)],
    [(0, 1, 9), (4, 7, 8), (2, 3, 11), (5, 10, 6)],
    [(9, 2, 1), (9, 11, 2), (9, 4, 11), (7, 11, 4), (5, 10, 6)],
    [(8, 4, 7), (3, 11, 5), (3, 5, 1), (5, 11, 6)],
    [(5, 1, 11), (5, 11, 6), (1, 0, 11), (7, 11, 4), (0, 4, 11)],
    [(0, 5, 9), (0, 6, 5), (0, 3, 6), (11, 6, 3), (8, 4, 7)],
    [(6, 5, 9), (6, 9, 11), (4, 7, 9), (7, 11, 9)],
    [(6, 4, 9), (9, 10, 6)],
    [(4, 10, 6), (4, 9, 10), (0, 8, 3)],
    [(10, 0, 1), (10, 6, 0), (6, 4, 0)],
    [(8, 3, 1), (8, 1, 6), (8, 6, 4), (6, 1, 10)],
    [(1, 4, 9), (1, 2, 4), (2, 6, 4)],
    [(3, 0, 8), (1, 2, 9), (2, 4, 9), (2, 6, 4)],
    [(0, 2, 4), (4, 2, 6)],
    [(8, 3, 2), (8, 2, 4), (4, 2, 6)],
    [(10, 4, 9), (10, 6, 4), (11, 2, 3)],
    [(0, 8, 2), (2, 8, 11), (4, 9, 10), (4, 10, 6)],
    [(3, 11, 2), (0, 1, 6), (0, 6, 4), (6, 1, 10)],
    [(6, 4, 1), (6, 1, 10), (4, 8, 1), (2, 1, 11), (8, 11, 1)],
    [(9, 6, 4), (9, 3, 6), (9, 1, 3), (11, 6, 3)],
    [(8, 11, 1), (8, 1, 0), (11, 6, 1), (9, 1, 4), (6, 4, 1)],
    [(3, 11, 6), (3, 6, 0), (0, 6, 4)],
    [(6, 4, 8), (8, 11, 6)],
    [(7, 10, 6), (7, 8, 10), (8, 9, 10)],
    [(0, 7, 3), (0, 10, 7), (0, 9, 10), (6, 7, 10)],
    [(10, 6, 7), (1, 10, 7), (1, 7, 8), (1, 8, 0)],
    [(10, 6, 7), (10, 7, 1), (1, 7, 3)],
    [(1, 2, 6), (1, 6, 8), (1, 8, 9), (8, 6, 7)],
    [(2, 6, 9), (2, 9, 1), (6, 7, 9), (0, 9, 3), (7, 3, 9)],
    [(7, 8, 0), (7, 0, 6), (6, 0, 2)],
    [(7, 3, 2), (6, 7, 2)],
    [(2, 3, 11), (10, 6, 8), (10, 8, 9), (8, 6, 7)],
    [(2, 0, 7), (2, 7, 11), (0, 9, 7), (6, 7, 10), (9, 10, 7)],
    [(1, 8, 0), (1, 7, 8), (1, 10, 7), (6, 7, 10), (2, 3, 11)],
    [(11, 2, 1), (11, 1, 7), (10, 6, 1), (6, 7, 1)],
    [(8, 9, 6), (8, 6, 7), (9, 1, 6), (11, 6, 3), (1, 3, 6)],
    [(0, 9, 1), (11, 6, 7)],
    [(7, 8, 0), (7, 0, 6), (3, 11, 0), (11, 6, 0)],
    [(7, 11, 6)],
    [(7, 6, 11)],
    [(3, 0, 8), (11, 7, 6)],
    [(0, 1, 9), (11, 7, 6)],
    [(8, 1, 9), (8, 3, 1), (11, 7, 6)],
    [(10, 1, 2), (6, 11, 7)],
    [(1, 2, 10), (3, 0, 8), (6, 11, 7)],
    [(2, 9, 0), (2, 10, 9), (6, 11, 7)],
    [(6, 11, 7), (2, 10, 3), (10, 8, 3), (10, 9, 8)],
    [(7, 2, 3), (6, 2, 7)],
    [(7, 0, 8), (7, 6, 0), (6, 2, 0)],
    [(2, 7, 6), (2, 3, 7), (0, 1, 9)],
    [(1, 6, 2), (1, 8, 6), (1, 9, 8), (8, 7, 6)],
    [(10, 7, 6), (10, 1, 7), (1, 3, 7)],
    [(10, 7, 6), (1, 7, 10), (1, 8, 7), (1, 0, 8)],
    [(0, 3, 7), (0, 7, 10), (0, 10, 9), (6, 10, 7)],
    [(7, 6, 10), (7, 10, 8), (8, 10, 9)],
    [(8, 4, 6), (6, 11, 8)],
    [(3, 6, 11), (3, 0, 6), (0, 4, 6)],
    [(8, 6, 11), (8, 4, 6), (9, 0, 1)],
    [(9, 4, 6), (9, 6, 3), (9, 3, 1), (11, 3, 6)],
    [(6, 8, 4), (6, 11, 8), (2, 10, 1)],
    [(1, 2, 10), (3, 0, 11), (0, 6, 11), (0, 4, 6)],
    [(4, 11, 8), (4, 6, 11), (0, 2, 9), (2, 10, 9)],
    [(10, 9, 3), (10, 3, 2), (9, 4, 3), (11, 3, 6), (4, 6, 3)],
    [(8, 2, 3), (8, 4, 2), (4, 6, 2)],
    [(0, 4, 2), (4, 6, 2)],
    [(1, 9, 0), (2, 3, 4), (2, 4, 6), (4, 3, 8)],
    [(1, 9, 4), (1, 4, 2), (2, 4, 6)],
    [(8, 1, 3), (8, 6, 1), (8, 4, 6), (6, 10, 1)],
    [(10, 1, 0), (10, 0, 6), (6, 0, 4)],
    [(4, 6, 3), (4, 3, 8), (6, 10, 3), (0, 3, 9), (10, 9, 3)],
    [(10, 9, 4), (4, 6, 10)],
    [(4, 9, 5), (7, 6, 11)],
    [(0, 8, 3), (4, 9, 5), (11, 7, 6)],
    [(5, 0, 1), (5, 4, 0), (7, 6, 11)],
    [(11, 7, 6), (8, 3, 4), (3, 5, 4), (3, 1, 5)],
    [(9, 5, 4), (10, 1, 2), (7, 6, 11)],
    [(6, 11, 7), (1, 2, 10), (0, 8, 3), (4, 9, 5)],
    [(7, 6, 11), (5, 4, 10), (4, 2, 10), (4, 0, 2)],
    [(3, 4, 8), (3, 5, 4), (3, 2, 5), (10, 5, 2), (11, 7, 6)],
    [(7, 2, 3), (7, 6, 2), (5, 4, 9)],
    [(9, 5, 4), (0, 8, 6), (0, 6, 2), (6, 8, 7)],
    [(3, 6, 2), (3, 7, 6), (1, 5, 0), (5, 4, 0)],
    [(6, 2, 8), (6, 8, 7), (2, 1, 8), (4, 8, 5), (1, 5, 8)],
    [(9, 5, 4), (10, 1, 6), (1, 7, 6), (1, 3, 7)],
    [(1, 6, 10), (1, 7, 6), (1, 0, 7), (8, 7, 0), (9, 5, 4)],
    [(4, 0, 10), (4, 10, 5), (0, 3, 10), (6, 10, 7), (3, 7, 10)],
    [(7, 6, 10), (7, 10, 8), (5, 4, 10), (4, 8, 10)],
    [(6, 9, 5), (6, 11, 9), (11, 8, 9)],
    [(3, 6, 11), (0, 6, 3), (0, 5, 6), (0, 9, 5)],
    [(0, 11, 8), (0, 5, 11), (0, 1, 5), (5, 6, 11)],
    [(6, 11, 3), (6, 3, 5), (5, 3, 1)],
    [(1, 2, 10), (9, 5, 11), (9, 11, 8), (11, 5, 6)],
    [(0, 11, 3), (0, 6, 11), (0, 9, 6), (5, 6, 9), (1, 2, 10)],
    [(11, 8, 5), (11, 5, 6), (8, 0, 5), (10, 5, 2), (0, 2, 5)],
    [(6, 11, 3), (6, 3, 5), (2, 10, 3), (10, 5, 3)],
    [(5, 8, 9), (5, 2, 8), (5, 6, 2), (3, 8, 2)],
    [(9, 5, 6), (9, 6, 0), (0, 6, 2)],
    [(1, 5, 8), (1, 8, 0), (5, 6, 8), (3, 8, 2), (6, 2, 8)],
    [(1, 5, 6), (2, 1, 6)],
    [(1, 3, 6), (1, 6, 10), (3, 8, 6), (5, 6, 9), (8, 9, 6)],
    [(10, 1, 0), (10, 0, 6), (9, 5, 0), (5, 6, 0)],
    [(0, 3, 8), (5, 6, 10)],
    [(5, 6, 10)],
    [(11, 5, 10), (7, 5, 11)],
    [(11, 5, 10), (11, 7, 5), (8, 3, 0)],
    [(5, 11, 7), (5, 10, 11), (1, 9, 0)],
    [(10, 7, 5), (10, 11, 7), (9, 8, 1), (8, 3, 1)],
    [(11, 1, 2), (11, 7, 1), (7, 5, 1)],
    [(0, 8, 3), (1, 2, 7), (1, 7, 5), (7, 2, 11)],
    [(9, 7, 5), (9, 2, 7), (9, 0, 2), (2, 11, 7)],
    [(7, 5, 2), (7, 2, 11), (5, 9, 2), (3, 2, 8), (9, 8, 2)],
    [(2, 5, 10), (2, 3, 5), (3, 7, 5)],
    [(8, 2, 0), (8, 5, 2), (8, 7, 5), (10, 2, 5)],
    [(9, 0, 1), (5, 10, 3), (5, 3, 7), (3, 10, 2)],
    [(9, 8, 2), (9, 2, 1), (8, 7, 2), (10, 2, 5), (7, 5, 2)],
    [(1, 3, 5), (3, 7, 5)],
    [(0, 8, 7), (0, 7, 1), (1, 7, 5)],
    [(9, 0, 3), (9, 3, 5), (5, 3, 7)],
    [(9, 8, 7), (5, 9, 7)],
    [(5, 8, 4), (5, 10, 8), (10, 11, 8)],
    [(5, 0, 4), (5, 11, 0), (5, 10, 11), (11, 3, 0)],
    [(0, 1, 9), (8, 4, 10), (8, 10, 11), (10, 4, 5)],
    [(10, 11, 4), (10, 4, 5), (11, 3, 4), (9, 4, 1), (3, 1, 4)],
    [(2, 5, 1), (2, 8, 5), (2, 11, 8), (4, 5, 8)],
    [(0, 4, 11), (0, 11, 3), (4, 5, 11), (2, 11, 1), (5, 1, 11)],
    [(0, 2, 5), (0, 5, 9), (2, 11, 5), (4, 5, 8), (11, 8, 5)],
    [(9, 4, 5), (2, 11, 3)],
    [(2, 5, 10), (3, 5, 2), (3, 4, 5), (3, 8, 4)],
    [(5, 10, 2), (5, 2, 4), (4, 2, 0)],
    [(3, 10, 2), (3, 5, 10), (3, 8, 5), (4, 5, 8), (0, 1, 9)],
    [(5, 10, 2), (5, 2, 4), (1, 9, 2), (9, 4, 2)],
    [(8, 4, 5), (8, 5, 3), (3, 5, 1)],
    [(0, 4, 5), (1, 0, 5)],
    [(8, 4, 5), (8, 5, 3), (9, 0, 5), (0, 3, 5)],
    [(9, 4, 5)],
    [(4, 11, 7), (4, 9, 11), (9, 10, 11)],
    [(0, 8, 3), (4, 9, 7), (9, 11, 7), (9, 10, 11)],
    [(1, 10, 11), (1, 11, 4), (1, 4, 0), (7, 4, 11)],
    [(3, 1, 4), (3, 4, 8), (1, 10, 4), (7, 4, 11), (10, 11, 4)],
    [(4, 11, 7), (9, 11, 4), (9, 2, 11), (9, 1, 2)],
    [(9, 7, 4), (9, 11, 7), (9, 1, 11), (2, 11, 1), (0, 8, 3)],
    [(11, 7, 4), (11, 4, 2), (2, 4, 0)],
    [(11, 7, 4), (11, 4, 2), (8, 3, 4), (3, 2, 4)],
    [(2, 9, 10), (2, 7, 9), (2, 3, 7), (7, 4, 9)],
    [(9, 10, 7), (9, 7, 4), (10, 2, 7), (8, 7, 0), (2, 0, 7)],
    [(3, 7, 10), (3, 10, 2), (7, 4, 10), (1, 10, 0), (4, 0, 10)],
    [(1, 10, 2), (8, 7, 4)],
    [(4, 9, 1), (4, 1, 7), (7, 1, 3)],
    [(4, 9, 1), (4, 1, 7), (0, 8, 1), (8, 7, 1)],
    [(4, 0, 3), (7, 4, 3)],
    [(4, 8, 7)],
    [(9, 11, 8), (11, 9, 10)],
    [(3, 0, 9), (3, 9, 11), (11, 9, 10)],
    [(0, 1, 10), (0, 10, 8), (8, 10, 11)],
    [(3, 1, 11), (11, 1, 10)],
    [(1, 2, 11), (1, 11, 9), (9, 11, 8)],
    [(3, 0, 9), (3, 9, 11), (1, 2, 9), (2, 11, 9)],
    [(0, 2, 11), (11, 8, 0)],
    [(3, 2, 11)],
    [(2, 3, 8), (2, 8, 10), (10, 8, 9)],
    [(2, 0, 9), (9, 10, 2)],
    [(2, 3, 8), (2, 8, 10), (0, 1, 8), (1, 10, 8)],
    [(10, 2, 1)],
    [(1, 3, 8), (9, 1, 8)],
    [(0, 9, 1)],
    [(3, 8, 0)],
    [],
]
# 立方体的8条边对于的始点和终点
LUT_EDGE_IDX_TO_START_END_POINTS_IDXS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def create_voxel_coords_grid(size_x, grid_size, size_y, size_z):
    x_ = np.linspace(-0.5 * size_x, 0.5 * size_x, grid_size)
    y_ = np.linspace(-0.5 * size_y, 0.5 * size_y, grid_size)
    z_ = np.linspace(-0.5 * size_z, 0.5 * size_z, grid_size)

    x, y, z = np.meshgrid(x_, y_, z_, indexing="ij")
    assert np.all(x[:, 0, 0] == x_)
    assert np.all(y[0, :, 0] == y_)
    assert np.all(z[0, 0, :] == z_)

    voxel_coordinates = np.stack([x, y, z], axis=-1)
    return voxel_coordinates


def create_artificial_sphere_sdf(voxel_coordinates, radius):
    voxel_dist_to_center = np.linalg.norm(voxel_coordinates, axis=-1, keepdims=True)

    # let's have a sdf, where at center of sphere sdf = 1, at border = 0, linear

    sdf_vals = radius - voxel_dist_to_center

    assert sdf_vals.shape[:-1] == voxel_coordinates.shape[:-1]
    assert sdf_vals.shape[-1] == 1
    return sdf_vals


def interpolate_crossing(loc_x1, sdf_val_at_x1, loc_x2, sdf_val_at_x2, thresh=0.0):
    # avoid divison by zero
    if sdf_val_at_x1 == sdf_val_at_x2:
        crossing_location = (loc_x2 + loc_x1) / 2
    else:
        crossing_location = loc_x1 + (thresh - sdf_val_at_x1) * (loc_x2 - loc_x1) / (sdf_val_at_x2 - sdf_val_at_x1)

    return crossing_location


def get_map_box_corner_idx_to_coords_tuple(x_idx, y_idx, z_idx):
    map_box_corner_idx_to_coords_tuple = {
        0: (x_idx, y_idx, z_idx),
        1: (x_idx, y_idx + 1, z_idx),
        2: (x_idx + 1, y_idx + 1, z_idx),
        3: (x_idx + 1, y_idx, z_idx),
        4: (x_idx, y_idx, z_idx + 1),
        5: (x_idx, y_idx + 1, z_idx + 1),
        6: (x_idx + 1, y_idx + 1, z_idx + 1),
        7: (x_idx + 1, y_idx, z_idx + 1),
    }

    assert len(map_box_corner_idx_to_coords_tuple) == 8, "you missed some cases or added too many"
    return map_box_corner_idx_to_coords_tuple


def marching_cubes(sdf_field, voxel_coords, thresh=0.0):
    assert sdf_field.shape[:-1] == voxel_coords.shape[:-1]
    triangle_vertices = []
    for x_idx in range(sdf_field.shape[0] - 1):
        for y_idx in range(sdf_field.shape[1] - 1):
            for z_idx in range(sdf_field.shape[2] - 1):

                map_box_corner_idx_to_coords_tuple = get_map_box_corner_idx_to_coords_tuple(x_idx, y_idx, z_idx)

                # 256 possible cases -> we need to match the correct case
                cube_lut_index = 0
                if sdf_field[map_box_corner_idx_to_coords_tuple[0]] < thresh:
                    cube_lut_index |= 1
                if sdf_field[map_box_corner_idx_to_coords_tuple[1]] < thresh:
                    cube_lut_index |= 2
                if sdf_field[map_box_corner_idx_to_coords_tuple[2]] < thresh:
                    cube_lut_index |= 4
                if sdf_field[map_box_corner_idx_to_coords_tuple[3]] < thresh:
                    cube_lut_index |= 8
                if sdf_field[map_box_corner_idx_to_coords_tuple[4]] < thresh:
                    cube_lut_index |= 16
                if sdf_field[map_box_corner_idx_to_coords_tuple[5]] < thresh:
                    cube_lut_index |= 32
                if sdf_field[map_box_corner_idx_to_coords_tuple[6]] < thresh:
                    cube_lut_index |= 64
                if sdf_field[map_box_corner_idx_to_coords_tuple[7]] < thresh:
                    cube_lut_index |= 128

                for edge_tuple_1, edge_tuple_2, edge_tuple_3 in LUT_CUBE_IDX_TO_TRIANGLES_EDGE_IDXS[cube_lut_index]:
                    vertices = []
                    for edge in (edge_tuple_1, edge_tuple_2, edge_tuple_3):
                        (
                            edge_start_point,
                            edge_end_point,
                        ) = LUT_EDGE_IDX_TO_START_END_POINTS_IDXS[edge]

                        start_point = voxel_coords[map_box_corner_idx_to_coords_tuple[edge_start_point]]
                        end_point = voxel_coords[map_box_corner_idx_to_coords_tuple[edge_end_point]]

                        vertex = interpolate_crossing(
                            start_point,
                            sdf_field[map_box_corner_idx_to_coords_tuple[edge_start_point]],
                            end_point,
                            sdf_field[map_box_corner_idx_to_coords_tuple[edge_end_point]],
                            thresh,
                        )
                        vertices.append(vertex)
                    triangle_vertices.append(vertices)
    triangle_vertices = np.array(triangle_vertices)

    return triangle_vertices.astype(np.float32)


def plot_mesh_colab(tri_vertices):
    faces = []
    triangle_vertices = tri_vertices.reshape((-1, 3))
    for i, t in enumerate(tri_vertices):
        faces.append([i * 3, i * 3 + 2, i * 3 + 1])

    mymesh = Trimesh(triangle_vertices, faces)

    mymesh.show(smooth=False)


def random_points_on_sphere(radius, num_points, center=np.array([0.0, 0.0, 0.0])):
    points = np.random.randn(num_points, 3)
    points /= np.linalg.norm(points, axis=-1, keepdims=True)
    points *= radius
    points += center

    assert points.shape == (num_points, 3)
    return points


def plot_points_colab(points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(points[::25, 0], points[::25, 1], points[::25, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]] * 3)

    plt.show()


def chamfer_distance(pcl_0, pcl_1):
    assert pcl_1.shape[-1] == 3
    assert pcl_0.shape[-1] == 3

    tree_0 = KDTree(pcl_0)
    tree_1 = KDTree(pcl_1)
    dist0, _ = tree_0.query(pcl_1)
    dist1, _ = tree_1.query(pcl_0)
    chamfer_dist = float(0.5 * (np.mean(dist0) + np.mean(dist1)))

    assert type(chamfer_dist) == float
    return chamfer_dist


if __name__ == '__main__':
    radius = 0.4

    size_x = 1.0  # size of our voxel grid
    size_y = 1.0
    size_z = 1.0

    test_grid_size = 8

    voxel_coordinates = create_voxel_coords_grid(size_x, test_grid_size, size_y, size_z)
    sdf_vals = create_artificial_sphere_sdf(voxel_coordinates, radius)
    triangle_vertices = marching_cubes(sdf_vals, voxel_coordinates, thresh=0.0)
    # plot_mesh_colab(triangle_vertices)

    gt_points = random_points_on_sphere(radius=radius, num_points=10000)

    # if you run locally use this (nicer, interactive plot)
    # plot_points_colab(gt_points)

    metrics = defaultdict(list)
    for grid_size in [8, 16, 32, 64, 128]:
        print("Processing grid size: {0}...".format(grid_size))
        voxel_coordinates = create_voxel_coords_grid(size_x, grid_size, size_y, size_z)

        sdf_vals = create_artificial_sphere_sdf(voxel_coordinates, radius)

        time_start = perf_counter()
        triangle_vertices = marching_cubes(sdf_vals, voxel_coordinates)
        runtime = perf_counter() - time_start
        triangle_vertex_centers = np.mean(triangle_vertices, axis=-2)

        metrics["Grid Size"].append(grid_size)
        metrics["time"].append(runtime)
        metrics["Chamfer Distance"].append(
            chamfer_distance(gt_points, triangle_vertex_centers)
        )
    print("Done!")

    fig, ax = plt.subplots()
    ax.plot(
        metrics["Grid Size"],
        metrics["Chamfer Distance"],
        label="Chamfer Distance",
        color="red",
    )
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Chamfer Distance [m]", color="red")
    plt.legend()
    ax2 = ax.twinx()
    ax2.plot(
        metrics["Grid Size"],
        metrics["time"],
        label="Execution time",
        color="blue",
        marker="o",
    )
    ax2.set_ylabel("Execution time [s]", color="blue")
    plt.legend()
    plt.show()
    print("Success!")
