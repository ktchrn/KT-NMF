import numpy as np
from scipy import sparse
from sklearn.neighbors import BallTree


def generate_nn_adjacency_matrix(data, N_nearest_neighbors=3):
	"""
	Generates an adjacency matrix in which each data point
	  is adjacent to the N_nearest_neighbors (Euclidean-)nearest 
	  data points in data space. 

	Arguments:	
		data					An N_features by N_dpts array

	Keyword arguments:
		N_nearest_neighbors 	This many nearest neighbors 
								  will be included in the 
								  adjacency matrix

	Returns:
		adjacency_matrix 		An N_dpts by N_dpts sparse array
	"""

	N_dpts = data.shape[1]
	adjacency_matrix = sparse.lil_matrix((N_dpts, N_dpts))

	#N_features x N_dpts is convenient for NMF algorithms.
	#sklearn prefers N_dpts x N_features
	_data_transpose = data.T

	tree = BallTree(_data_transpose)
    
    #N_nearest_neighbors+1 because the nearest neighbor 
    #will be the point itself
	ids_nearest = tree.query(_data_transpose, k=N_nearest_neighbors+1, return_distance=False, dualtree=True, sort_results=True)

	adjacency_matrix[ids_nearest[:,0], ids_nearest[:,1:].T] = 1

	#Dot products in GRNMF end up operating on columns rather than on rows,
	#CSC is ~1.5 times faster than CSR
	adjacency_matrix = adjacency_matrix.tocsc()

	return adjacency_matrix


def generate_2d_spatial_adjacency_matrix(data, N_x, N_y, max_pix=2, pix_scale=2.):
	"""

	"""

	N_dpts = N_x*N_y

	adjacency_matrix = sparse.lil_matrix((N_dpts, N_dpts))
	base_x, base_y = np.meshgrid(np.arange(N_x), np.arange(N_y))
	base_x = base_x.flatten()
	base_y = base_y.flatten()

	for dx in range(max_pix):
		for dy in range(max_pix):
			if (dx==0) and (dy==0):
				pass
			else:
				add_shifted_weight(adjacency_matrix, (base_x, base_y),
				   (dx, dy),
				   N_x, N_y,
				   weight=np.exp(-0.5*(dx**2 + dy**2)/pix_scale**2),
				   symmetric=True)
	adjacency_matrix = adjacency_matrix.tocsc()
	return adjacency_matrix


def flatten_2d_index(ix, iy, N_y):
    return ix*N_y + iy
    
    
def add_shifted_weight(adjacency_matrix, base, shift, N_x, N_y, weight=1.,
                       symmetric=True):
    shifted_x = base[0] + shift[0]
    shifted_y = base[1] + shift[1]

    within_bounds = (shifted_x < N_x)
    within_bounds *= (shifted_x >= 0)
    within_bounds *= (shifted_y < N_y)
    within_bounds *= (shifted_y >= 0)

    base_1d = flatten_2d_index(base[0][within_bounds],
                               base[1][within_bounds],
                               N_y)
    shifted_1d = flatten_2d_index(shifted_x[within_bounds],
                                  shifted_y[within_bounds],
                                  N_y)
    adjacency_matrix[base_1d, shifted_1d] = weight
    if symmetric:
        adjacency_matrix[shifted_1d, base_1d] = weight


















