import numpy as np
from scipy import sparse


def order_by_first_moment(basis, coeffs):
    """

    """

    N_features = basis.shape[0]
    first_moment = basis*np.arange(N_features)[:,None]
    first_moment = np.sum(first_moment, axis=0)/np.sum(basis, axis=0)
    order = np.argsort(first_moment)
    ord_basis = basis[:,order]
    ord_coeffs = coeffs[:,order]
    return ord_basis, ord_coeffs


def order_by_projected_power(basis, coeffs):
    """

    """

    projections = np.einsum('jk,ik->k', coeffs, basis)
    order = np.argsort(projections)
    ord_basis = basis[:,order]
    ord_coeffs = coeffs[:,order]
    return ord_basis, ord_coeffs
    

class GRNMF(object):
    """

    """

    def __init__(self, data, graph_regularization_matrix, rank=2):
        self.data = data
        self.W = graph_regularization_matrix
        self.L, self.D = sparse.csgraph.laplacian(self.W, 
            return_diag=True, 
            normed=False)

        self.rank = rank
        self.N_features, self.N_dpts = self.data.shape

        self.basis = np.abs(np.random.rand(self.N_features, self.rank))
        self.coeffs = np.abs(np.random.rand(self.N_dpts, self.rank))

        self.lamb = None


    def run(self, lamb=0., maxiter=500, tol=0.1, silent=True):
        """

        """
        last_obj = np.Infinity
        not_converged = True
        under_maxiter = True

        self.lamb = lamb

        iter_i = 0

        if not silent:
            print "Starting factorization"
        while not_converged and under_maxiter:
            iter_i += 1
            self.GRNMF_step(lamb)
            new_obj = self.GRNMF_objective(lamb)
            if (last_obj - new_obj) < tol:
                not_converged = False
            last_obj = new_obj

            if iter_i > maxiter:
                under_maxiter = False

        basis_norm = np.sqrt((self.basis**2).sum(axis=0))
        self.basis /= basis_norm
        self.coeffs *= basis_norm
        
        if under_maxiter:
            print "Converged in {:.0f} iterations".format(iter_i)
        elif not_converged:
            print "Didn't converge in {:.0f} iterations".format(iter_i)


    def GRNMF_step(self, lamb):
        new_basis = self.basis*np.dot(self.data,self.coeffs)
        new_basis /= np.dot(self.basis,np.dot(self.coeffs.T,self.coeffs))
        self.basis = new_basis

        if lamb == 0:
            new_coeffs = self.coeffs*np.dot(self.data.T, self.basis)
            new_coeffs /= np.dot(self.coeffs, np.dot(self.basis.T, self.basis))
        else:
            new_coeffs = self.coeffs*(np.dot(self.data.T, self.basis) + lamb*self.W.dot(self.coeffs))
            new_coeffs /= (np.dot(self.coeffs, np.dot(self.basis.T, self.basis)) + lamb*np.dot(self.D, self.coeffs))
        self.coeffs = new_coeffs
    
    
    def GRNMF_objective(self, lamb):
        discrepancy = self.data - np.dot(self.basis, self.coeffs.T)
        obj = np.trace(np.dot(discrepancy, discrepancy.T))
    
        if lamb>0:
            obj += lamb*np.trace(np.dot(self.coeffs.T, self.L.dot(self.coeffs)))
        return obj