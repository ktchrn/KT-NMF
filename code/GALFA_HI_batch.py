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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_fname')
    args = parser.parse_args()

    silent = False

    with open(args.batch_fname, 'r') as file:
        data_fname, argsets = pickle.load(file)

    if not silent:
        print "Preparing data"

    with fits.open(data_fname) as fits_file:
        #hard convert to double to avoid bit order issues
        data = np.array(fits_file[0].data, dtype=np.double)

    preprocessed_data, N_features, spatial_shape = cube_pipeline.preprocess_data(data, True)

    for argset in argsets:
        process_single_parset(preprocessed_data, spatial_shape, *argset)

