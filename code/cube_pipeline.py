from astropy.io import fits
import cPickle as pickle
import GRNMF
import matplotlib.pyplot as plt
import numpy as np


def preprocess_data(input_cube, normalize=False):
	"""
	Description

	Arguments:

	Keyword arguments:

	Returns:

	"""

	N_features, N_x, N_y = input_cube.shape
	spatial_shape = [N_x, N_y]

	preprocessed_data = input_cube.reshape(N_features, N_x*N_y)
	preprocessed_data = np.nan_to_num(preprocessed_data)
	preprocessed_data = np.clip(preprocessed_data, 0, np.Infinity)

	if normalize:
		preprocessed_data /= preprocessed_data.sum()

	return preprocessed_data, N_features, spatial_shape


def get_summary(GRNMF_object, spatial_shape, 
		adjacency_descriptor, order_function=None):
	"""
	Description

	Arguments:

	Keyword arguments:

	Returns:

	"""

	if order_function is None:
		coeffs = GRNMF_object.coeffs
		basis = GRNMF_object.basis
	else:
		basis, coeffs = order_function(GRNMF_object.basis, 
			GRNMF_object.coeffs)

	summary = [basis, coeffs, spatial_shape, adjacency_descriptor, 
		GRNMF_object.rank, GRNMF_object.lamb]
	return summary


def save_summary(GRNMF_object, spatial_shape, 
		adjacency_descriptor, 
		order_function=None,
		summaryfile_prefix='./', 
		summaryfile_suffix='_summary.pi',
		pickle_protocol=-1):
	"""
	Description

	Arguments:

	Keyword arguments:

	Returns:

	"""
	
	summary = get_summary(GRNMF_object, spatial_shape, 
		adjacency_descriptor, order_function=order_function)

	savename = '{}_adjType_{}_rank{:.0f}_lambda{:.0f}{}'.format(summaryfile_prefix, 
		adjacency_descriptor, GRNMF_object.rank, GRNMF_object.lamb, summaryfile_suffix)

	with open(savename, 'w') as file:
		pickle.dump(summary, file)	

	return summary


def save_basis_to_fits(GRNMF_object, savefile_prefix='./', savefile_fullname=None):
	"""
	Description

	Arguments:

	Keyword arguments:

	Returns:

	"""
	pass


def save_coefficients_to_fits(GRNMF_object, spatial_shape, 
		savefile_prefix='./', savefile_fullname=None):
	"""
	Description

	Arguments:

	Keyword arguments:

	Returns:

	"""
	pass


def plot_summary(summary, plot_type='coeffs', save=True,
		figfile_prefix='./',
		figfile_suffix='.png'):
    """

    """

    if type(summary) is str:
    	with open(summary, 'r') as file:
    		basis, coeffs, spatial_shape, adjacency_descriptor, rank, lamb = pickle.load(file)
    else:
	    basis, coeffs, spatial_shape, adjacency_descriptor, rank, lamb = summary

    N_cols = np.ceil(np.sqrt(rank))
    N_rows = N_cols + 1

    plt.figure(figsize=[10,10])

    if plot_type == 'coeffs':
        images = coeffs.reshape([spatial_shape[0], 
            spatial_shape[1], rank])
        vmax = None
    else:
        images = np.einsum('jk,ik->jk', coeffs, basis)
        images = images.reshape([spatial_shape[0], 
            spatial_shape[1], rank])
        vmax = images.max()


    for rank_i in range(rank):
        ax = plt.subplot(N_rows, N_cols, rank_i+1)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(images[:,:,rank_i], cmap='gray_r', aspect='auto', 
            interpolation='nearest', origin='lower',
            vmin=0, vmax=vmax)

        plt.text(0.1, 0.9, r'$\bf{:.0f}$'.format(rank_i), color='red',
                 transform=ax.transAxes, fontsize=16, ha='center', va='center')

    ax = plt.subplot(N_rows, 1, N_rows)
    plt.imshow(basis.T, interpolation='nearest', cmap='gray_r', aspect='auto')

    plt.xlabel('Velocity slice')
    plt.ylabel('Latent feature index')

    if plot_type == 'coeffs':
    	plt.suptitle('Basis coefficients')
    else:
    	plt.suptitle('Projected power')

    try:
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
    except:
        plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05)

    if save:
    	savename = '{}_adjType_{}_rank{:.0f}_lambda{:.0f}_{}{}'.format(figfile_prefix, 
			adjacency_descriptor, rank, lamb, 
			plot_type, figfile_suffix)
    	plt.savefig(savename)