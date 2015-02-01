import adjacency
import argparse
from astropy.io import fits
import cPickle as pickle
import cube_pipeline
import GRNMF
import numpy as np
    

def process_single_parset(preprocessed_data, spatial_shape, W_type, W_kwargs, K, lamb, maxiter, tol):
    silent = False
    #Build the adjancency matrix W
    if not silent:
        print "Bulding adjacency matrix"

    if W_type == 'spatial':
        W = adjacency.generate_2d_spatial_adjacency_matrix(preprocessed_data, *spatial_shape,
            **W_kwargs)
        adjacency_type = '{}_mp{}_ps{}'.format(W_type, 
            W_kwargs['max_pix'], W_kwargs['pix_scale'])
    elif W_type == 'nearest_neighbor':
        W = adjacency.generate_nn_adjacency_matrix(preprocessed_data, **W_kwargs)
        adjacency_type = '{}_Nnn{}'.format(W_type, W_kwargs['N_nearest_neighbors'])
    elif W_type == 'None':
        W = None
        adjacency_type = 'none'
        
    #Do GRNMF
    if not silent:
        print "Factorizing"

    instance = GRNMF.GRNMF(preprocessed_data, W, rank=K)
    instance.run(lamb=lamb, maxiter=maxiter, tol=tol, silent=silent)

    summary = cube_pipeline.save_summary(instance, spatial_shape, adjacency_type, 
        summaryfile_prefix='../data/GALFA_HI_ex', 
        order_function=GRNMF.order_by_first_moment,
        )
    cube_pipeline.plot_summary(summary,
            figfile_prefix='../figs/GALFA_HI_ex',
            figfile_suffix='.png',
            plot_type='coeffs',
            save=True)
    cube_pipeline.plot_summary(summary,
            figfile_prefix='../figs/GALFA_HI_ex',
            figfile_suffix='.png',
            plot_type='projection',
            save=True)


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




    
