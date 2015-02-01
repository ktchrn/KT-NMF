import adjacency
import argparse
from astropy.io import fits
import cPickle as pickle
import cube_pipeline
import GRNMF
import numpy as np
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_fname')
    args = parser.parse_args()

    silent = False
    n_batch = 2
    batch_stem = './test_batch_{}.pi'
    W_types = ['None', 'spatial', 'nearest_neighbor']
    K_vals = [16, 25, 36]
    lamb_vals = [100., 1000., 10000.]
    maxiter = 5
    N_nearest = 6
    max_pix = 10
    pix_scale = 6.
    tol = 0.

    spatial_kwargs = {'max_pix':max_pix, 'pix_scale':pix_scale}
    nn_kwargs = {'N_nearest_neighbors':N_nearest}
    none_kwargs = {}
    W_kwargs = [none_kwargs, spatial_kwargs, nn_kwargs]

    argsets = []
    for W_type, W_kwarg in zip(W_types, W_kwargs):
        for K_val in K_vals:
            if W_type == 'None':
                argsets.append([W_type, W_kwarg, K_val, 0, maxiter, tol])
            else:
                for lamb in lamb_vals:
                    argsets.append([W_type, W_kwarg, K_val, lamb, maxiter, tol])


    for batch_i in range(n_batch):
        batch = [args.data_fname, argsets[batch_i::n_batch]]
        batch_name = batch_stem.format(batch_i)
        with open(batch_name, 'w') as file:
            pickle.dump(batch, file)




    
