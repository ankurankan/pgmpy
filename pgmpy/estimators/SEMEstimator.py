import numpy as np
import pandas as pd
import torch

from pgmpy.models import SEM
from pgmpy.data import Data
from pgmpy.global_vars import device, dtype
from pgmpy.utils import optimize, pinverse


class SEMEstimator(object):
    """
    Base class of SEM estimators. All the estimators inherit this class.
    """
    def __init__(self, model):
        if not isinstance(model, SEM):
            raise ValueError("model should be an instance of SEM class. Got type: {t}".format(t=type(model)))

        self.model = model

        # Initialize mask tensors
        masks = self.model.get_masks()
        fixed_masks = self.model.get_fixed_masks()
        self.masks = {}
        self.fixed_masks = {}
        model_params = ['B', 'gamma', 'wedge_y', 'wedge_x', 'phi', 'theta_e', 'theta_del', 'psi']
        for i, model_param in enumerate(model_params):
            self.masks[model_param] = torch.tensor(masks[i], device=device, dtype=dtype, requires_grad=False)
            self.fixed_masks[model_param] = torch.tensor(fixed_masks[i], device=device, dtype=dtype,
                                                         requires_grad=False)
        self.B_eye = torch.eye(self.masks['B'].shape[0], device=device, dtype=dtype, requires_grad=False)

    def _get_implied_cov(self, B, gamma, wedge_y, wedge_x, phi, theta_e, theta_del, psi):
        """
        Computes the implied covariance matrix from the given parameters.
        """
        B_masked = (torch.mul(B, self.masks['B']) + self.fixed_masks['B'])
        B_inv = pinverse(self.B_eye - B_masked)
        gamma_masked = torch.mul(gamma, self.masks['gamma']) + self.fixed_masks['gamma']
        wedge_y_masked = torch.mul(wedge_y, self.masks['wedge_y']) + self.fixed_masks['wedge_y']
        wedge_x_masked = torch.mul(wedge_x, self.masks['wedge_x']) + self.fixed_masks['wedge_x']
        phi_masked = torch.mul(phi, self.masks['phi']) + self.fixed_masks['phi']
        theta_e_masked = torch.mul(theta_e, self.masks['theta_e']) + self.fixed_masks['theta_e']
        theta_del_masked = torch.mul(theta_del, self.masks['theta_del']) + self.fixed_masks['theta_del']
        psi_masked = torch.mul(psi, self.masks['psi']) + self.fixed_masks['psi']

        sigma_yy = wedge_y_masked @ B_inv @ (gamma_masked @ phi_masked @ gamma_masked.t() + psi_masked) @ \
                   B_inv.t() @ wedge_y_masked.t() + theta_e_masked
        sigma_yx = wedge_y_masked @ B_inv @ gamma_masked @ phi_masked @ wedge_x_masked.t()
        sigma_xy = sigma_yx.t()
        sigma_xx = wedge_x_masked @ phi_masked @ wedge_x_masked.t() + theta_del_masked

        sigma = torch.cat((torch.cat((sigma_yy, sigma_yx), 1), torch.cat((sigma_xy, sigma_xx), 1)), 0)
        return sigma

    def ml_loss(self, params, loss_args):
        """
        Method to compute the Maximum Likelihood loss function. The optimizer calls this
        method after each iteration with updated params to compute the new loss.

        The fitting function for ML is:
            $$ F_{ML} = \log |\Sigma(\theta)| + tr(S \Sigma^{-1}(\theta)) - \log S - (p+q) $$

        Parameters
        ----------
        params: dict
            params contain all the variables which are updated in each iteration of the
            optimization.

        loss_args: dict
            loss_args contain all the variable which are not updated in each iteration but
            are required to compute the loss.

        Returns
        -------
        torch.tensor: The loss value for the given params and loss_args
        """
        S = loss_args['S']
        sigma = self._get_implied_cov(params['B'], params['gamma'], params['wedge_y'],
                                      params['wedge_x'], params['phi'], params['theta_e'],
                                      params['theta_del'], params['psi'])

        return (sigma.det().clamp(min=1e-4).log() + (S @ pinverse(sigma)).trace() - S.logdet() -
                (len(self.model.y)+ len(self.model.x)))

    def uls_loss(self, params, loss_args):
        """
        Method to compute the Unweighted Least Squares fitting function. The optimizer calls
        this method after each iteration with updated params to compute the new loss.

        The fitting function for ML is:
            $$ F_{ULS} = tr[(S - \Sigma(\theta))^2] $$

        Parameters
        ----------
        params: dict
            params contain all the variables which are updated in each iteration of the
            optimization.

        loss_args: dict
            loss_args contain all the variable which are not updated in each iteration but
            are required to compute the loss.

        Returns
        -------
        torch.tensor: The loss value for the given params and loss_args
        """
        S = loss_args['S']
        sigma = self._get_implied_cov(params['B'], params['gamma'], params['wedge_y'],
                                      params['wedge_x'], params['phi'], params['theta_e'],
                                      params['theta_del'], params['psi'])
        return (S - sigma).pow(2).trace()

    def gls_loss(self, params, loss_args):
        """
        Method to compute the Weighted Least Squares fitting function. The optimizer calls
        this method after each iteration with updated params to compute the new loss.

        The fitting function for ML is:
            $$ F_{ULS} = tr \{ [(S - \Sigma(\theta)) W^{-1}]^2 \} $$

        Parameters
        ----------
        params: dict
            params contain all the variables which are updated in each iteration of the
            optimization.

        loss_args: dict
            loss_args contain all the variable which are not updated in each iteration but
            are required to compute the loss.

        Returns
        -------
        torch.tensor: The loss value for the given params and loss_args
        """
        S = loss_args['S']
        W_inv = pinverse(loss_args['W'])
        sigma = self._get_implied_cov(params['B'], params['gamma'], params['wedge_y'],
                                      params['wedge_x'], params['phi'], params['theta_e'],
                                      params['theta_del'], params['psi'])
        return ((S - sigma) @ W_inv).pow(2).trace()

    def fit(self, data, method, opt='adam', exit_delta=1e-4, max_iter=1000, **kwargs):
        """
        Estimate the parameters of the model from the data.

        Parameters
        ----------
        data: pandas DataFrame or pgmpy.data.Data instance
            The data from which to estimate the parameters of the model.

        method: str ("ml"|"uls"|"gls"|"2sls")
            The fitting function to use.
            ML : Maximum Likelihood
            ULS: Unweighted Least Squares
            GLS: Generalized Least Squares
            2sls: 2-SLS estimator

        **kwargs: dict
            Extra parameters required in case of some estimators.
            GLS:
                W: np.array (n x n) where n is the number of observe variables.
            2sls:
                x:
                y:

        Returns
        -------
            pgmpy.model.SEM instance: Instance of the model with estimated parameters
        """
        # Check if given arguements are valid
        if not isinstance(data, (pd.DataFrame, Data)):
            raise ValueError("data must be a pandas DataFrame. Got type: {t}".format(t=type(data)))

        if not sorted(data.columns) == sorted(self.model.observed):
            raise ValueError("""The column names data do not match the variables in the model. Expected:
                                {expected}. Got: {got}""".format(expected=sorted(self.model.observed),
                                                                 got=sorted(data.columns)))

        # Initialize random values as the initial values for the optimization
        B = torch.rand(*self.masks['B'].shape, device=device, dtype=dtype, requires_grad=True)
        gamma = torch.rand(*self.masks['gamma'].shape, device=device, dtype=dtype, requires_grad=True)
        wedge_y = torch.rand(*self.masks['wedge_y'].shape, device=device, dtype=dtype, requires_grad=True)
        wedge_x = torch.rand(*self.masks['wedge_x'].shape, device=device, dtype=dtype, requires_grad=True)
        phi = torch.rand(*self.masks['phi'].shape, device=device, dtype=dtype, requires_grad=True)
        theta_e = torch.rand(*self.masks['theta_e'].shape, device=device, dtype=dtype, requires_grad=True)
        theta_del = torch.rand(*self.masks['theta_del'].shape, device=device, dtype=dtype, requires_grad=True)
        psi = torch.rand(*self.masks['psi'].shape, device=device, dtype=dtype, requires_grad=True)

        # Compute the covariance of the data
        variable_order = self.model.y + self.model.x
        S = data.cov().reindex(variable_order, axis=1).reindex(variable_order, axis=0)
        S = torch.tensor(S.values, device=device, dtype=dtype, requires_grad=False)

        # Optimize the parameters
        if method.lower() == 'ml':
            params = optimize(self.ml_loss, params={'B': B, 'gamma': gamma, 'wedge_y': wedge_y,
                                                   'wedge_x': wedge_x, 'phi': phi, 'theta_e':
                                                   theta_e, 'theta_del': theta_del, 'psi': psi},
                            loss_args={'S': S}, opt=opt, exit_delta=exit_delta, max_iter=max_iter)
            for key, value in params.items():
                params[key] = value * self.masks[key] + self.fixed_masks[key]

        elif method.lower() == 'uls':
            params = optimize(self.uls_loss, params={'B': B, 'gamma': gamma, 'wedge_y': wedge_y,
                                                     'wedge_x': wedge_x, 'phi': phi, 'theta_e':
                                                     theta_e, 'theta_del': theta_del, 'psi': psi},
                              loss_args={'S': S}, opt=opt, exit_delta=exit_delta, max_iter=max_iter)

            for key, value in params.items():
                params[key] = value * self.masks[key] + self.fixed_masks[key]

        elif method.lower() == 'gls':
            W = torch.tensor(kwargs['W'], device=device, dtype=dtype, requires_grad=False)
            params = optimize(self.gls_loss, params={'B': B, 'gamma': gamma, 'wedge_y': wedge_y,
                                                     'wedge_x': wedge_x, 'phi': phi, 'theta_e':
                                                     theta_e, 'theta_del': theta_del, 'psi': psi},
                              loss_args={'S': S, 'W': W}, opt=opt, exit_delta=exit_delta,
                              max_iter=max_iter)

            for key, value in params.items():
                params[key] = value * self.masks[key] + self.fixed_masks[key]

        elif method.lower() == '2sls' or method.lower() == '2-sls':
            raise NotImplementedError("2-SLS is not implemented yet")

        # Compute goodness of fit statistics.
        N = data.shape[0]
        sample_cov = S.detach().numpy()
        sigma_hat = self._get_implied_cov(params['B'], params['gamma'], params['wedge_y'], params['wedge_x'],
                                          params['phi'], params['theta_e'], params['theta_del'],
                                          params['psi']).detach().numpy()
        residual = sample_cov - sigma_hat

        norm_residual = np.zeros(residual.shape)
        for i in range(norm_residual.shape[0]):
            for j in range(norm_residual.shape[1]):
                norm_residual[i, j] = (sample_cov[i, j] - sigma_hat[i, j]) / np.sqrt(
                                      ((sigma_hat[i, i] * sigma_hat[j, j]) + (sigma_hat[i, j]**2)) / N)

        # Compute chi-square value.
        likelihood_ratio = -(N-1) * (np.log(np.linalg.det(sigma_hat)) + (np.linalg.inv(sigma_hat) @ S).trace() -
                                     np.log(np.linalg.det(S)) - S.shape[0])
        if method.lower() == 'ml':
            error = self.ml_loss(params, loss_args={'S': S})
        elif method.lower() == 'uls':
            error = self.uls_loss(params, loss_args={'S': S})
        elif method.lower() == 'gls':
            error = self.gls_loss(params, loss_args={'S': S, 'W': W})
        chi_square = likelihood_ratio / error.detach().numpy()

        # TODO: Compute the degree of freedom.

        summary = {'Sample Size': N,
                   'Sample Covariance': sample_cov,
                   'Model Implied Covariance': sigma_hat,
                   'Residual': residual,
                   'Normalized Residual': norm_residual,
                   'chi_square': chi_square,
                  }

        # Update the model with the learned params
        self.model.set_params({key: value.detach().numpy() for key, value in params.items()})

        return summary
