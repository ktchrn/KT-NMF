import adjacency
from astropy.io import fits
import cube_pipeline
import numpy as np
import GRNMF
    

if __name__=='__main__':
    silent = False
    W_type = 'spatial'
    data_dir = '/Users/K/Downloads/'
    data_fname = 'Smallified_LLCC.fits'
    K = 36
    lamb = 5000.
    maxiter = 1000
    N_nearest = 4
    max_pix = 10
    pix_scale = 6.
    tol = 1.e-12

    if not silent:
        print "Preparing data"

    fname = data_dir + data_fname
    with fits.open(fname) as fits_file:
        #hard convert to double to avoid bit order issues
        data = np.array(fits_file[0].data, dtype=np.double)

    preprocessed_data, N_features, spatial_shape = cube_pipeline.preprocess_data(data, True)

    #Build the adjancency matrix W
    if not silent:
        print "Bulding adjacency matrix"

    if W_type == 'spatial':
        W = adjacency.generate_2d_spatial_adjacency_matrix(preprocessed_data, *spatial_shape, 
            max_pix=max_pix, pix_scale=pix_scale)
        adjacency_type = '{}_mp{}_ps{}'.format(W_type, max_pix, pix_scale)
    elif W_type == 'nearest_neighbor':
        W = adjacency.generate_nn_adjacency_matrix(preprocessed_data, N_nearest_neighbors=N_nearest)
        adjacency_type = '{}_Nnn{}'.format(N_nearest)
        
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







    
