import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel

def _ll_latentnorm(y, X, beta, alph):
    mu = (np.dot(X, beta))
    sigma = np.exp(np.dot(X, alph))
    z_bar = (np.log1p(y) - mu) / sigma 
    z_underbar = (np.log(y) - mu) / sigma
    ll = stats.norm.logcdf(z_underbar) + np.log(np.exp(stats.norm.logcdf(z_bar) - stats.norm.logcdf(z_underbar)) - 1.)
    return ll

def _get_kappa(z_bar, z_underbar, q):
    # Log denominator
    log_den = stats.norm.logcdf(z_underbar) + np.log(np.exp(stats.norm.logcdf(z_bar) - stats.norm.logcdf(z_underbar)) - 1.)

    # Witchcraft to get the numerator
    z_bar_pos = ((z_bar)**q) > 0
    z_underbar_pos = ((z_underbar)**q) > 0
    both_pos = z_bar_pos & z_underbar_pos
    both_neg = (~z_bar_pos) & (~z_underbar_pos)
    sign_switch = z_bar_pos & (~z_underbar_pos)
    log_mod_bar = q * np.log(np.abs(z_bar)) + stats.norm.logpdf(z_bar)
    log_mod_underbar = q * np.log(np.abs(z_underbar)) + stats.norm.logpdf(z_underbar)
    bar_mod_larger = log_mod_bar > log_mod_underbar
    keep_order = sign_switch | (both_pos & bar_mod_larger) | (both_neg & (~bar_mod_larger))
    sign = (-1)**(~keep_order)
    log_mod_b = keep_order * log_mod_bar + (~keep_order) * log_mod_underbar
    log_mod_a = (~keep_order) * log_mod_bar + keep_order * log_mod_underbar
    mod_b_greater = log_mod_b > log_mod_a
    log_mod_c = (~mod_b_greater) * log_mod_b + mod_b_greater * log_mod_a
    log_mod_d = mod_b_greater * log_mod_b + (~mod_b_greater) * log_mod_a
    const = (-1)**(~sign_switch)
    log_num = log_mod_c + np.log(np.exp(log_mod_d - log_mod_c) + const)

    # Create kappa_q
    kappa_q = sign * np.exp(log_num - log_den)
    return kappa_q

def _gradutils(y, X, beta, alph):
    mu = (np.dot(X, beta))
    sigma = np.exp(np.dot(X, alph))

    z_bar = (np.log1p(y) - mu) / sigma 
    z_underbar = (np.log(y) - mu) / sigma
    
    kappa_0 = _get_kappa(z_bar, z_underbar, 0)
    kappa_1 = _get_kappa(z_bar, z_underbar, 1)
    kappa_2 = _get_kappa(z_bar, z_underbar, 2)
    kappa_3 = _get_kappa(z_bar, z_underbar, 3)
    
    return kappa_0, kappa_1, kappa_2, kappa_3, mu, sigma
    
def _vec_matrix_multiply(a, B):
    return np.apply_along_axis(lambda x: x * a, 0, B)

def _em_gradutils(W, sigma, c, alpha, return_hessian=False):
    sigma_neg_2 = sigma**-2
    grad = W.T @ (sigma_neg_2 * c - 1.) - alpha
    hessian = None
    if return_hessian:
        W_sqrt_k = _vec_matrix_multiply(np.sqrt(c)/sigma, W)
        hessian = -2. * W_sqrt_k.T @ W_sqrt_k
    return grad, hessian

class DiscreteLognormal(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(DiscreteLognormal, self).__init__(endog, exog, **kwds)
        self.nparams = 22
        
    def nloglikeobs(self, params):
        exog = self.exog
        endog = self.endog
        beta = params[:11] #first 11 are for mu
        alph = params[11:] #last 11 are for sigma
        ll = _ll_latentnorm(endog, exog, beta, alph)
        params_alt = params.copy()
        params_alt[0] = 0.
        return -ll - self.penalty*np.sum(params_alt**2)/self.endog.size
    
    def score(self, params):
        y = self.endog
        X = self.exog
        beta = params[:11] #first 11 are for mu
        alph = params[11:] #last 11 are for sigma
        
        kappa_0, kappa_1, kappa_2, kappa_3, mu, sigma = _gradutils(y, X, beta, alph)
        
        beta_alt = beta.copy()
        beta_alt[0] = 0
        
        grad_beta = -(kappa_0 / sigma) @ X - self.penalty*2 * beta_alt
        grad_alph = -kappa_1 @ X - self.penalty*2 * alph
        
        return np.append(grad_beta, grad_alph)
    
    def hessian(self, params):
        y = self.endog
        X = self.exog
        beta = params[:11] #first 11 are for mu
        alph = params[11:] #last 11 are for sigma
        
        kappa_0, kappa_1, kappa_2, kappa_3, mu, sigma = _gradutils(y, X, beta, alph)
        
        k_beta = (kappa_0**2 + kappa_1) / sigma**2
        k_alph = kappa_1 * (kappa_1 - 1) + kappa_3
        k_beta_alph = (kappa_2 + kappa_0*(kappa_1 - 1)) / sigma
        H_beta = np.zeros([11, 11])
        H_alph = np.zeros([11, 11])
        H_beta_alph = np.zeros([11, 11])
                  
        for i in range(X.shape[0]):
            x = X[i]
            xxT = np.outer(x, x)
            H_beta -= k_beta[i] * xxT
            H_alph -= k_alph[i] * xxT
            H_beta_alph -= k_beta_alph[i] * xxT
        
        H_all = np.block([[H_beta, H_beta_alph], [H_beta_alph.T, H_alph]]) # 22 x 22
        penalty_matrix = self.penalty*2 * np.eye(22)
        penalty_matrix[0, 0] = 0.

        return H_all - penalty_matrix
    
    def predict(self, params, exog=None, n=1000, return_variance=False):
        if exog is None:
            X = self.exog
        else:
            X = exog
            
        beta = params[:11] #first 11 are for mu
        alph = params[11:] #last 11 are for sigma
        mu = (np.dot(X, beta))
        sigma = np.exp(np.dot(X, alph))
        z = stats.norm(mu, sigma).rvs(size = (n, self.endog.size)) # n random realizations. Could find closed form..
        y = np.floor(np.exp(z))
        
        if return_variance:
            var_y = np.var(y, axis=0)
            return var_y
        else:
            mean_y = np.mean(y, axis=0)
            return mean_y

    
    def mse(self):
        r = self.endog - self.predict()
        return np.mean(r**2)
    
    
    def fit(self, start_params=None, method="EM", maxiter=100, use_hessian=False, step_size=1e-4, tol=1e-6, maxfun=5000, penalty=0., **kwds):
        self.penalty = penalty
        if start_params is None:
            # Reasonable starting values
            start_params = np.zeros(self.nparams)
            start_params[0] = np.log(np.mean(self.endog)) # beta intercept

        if method == "EM":  
#             print("Using EM algorithm")
            self.em(
                start_params=start_params,
                maxiter=maxiter,
                use_hessian=use_hessian,
                step_size=step_size,
                tol=tol)
            return self

        else:
            return super(DiscreteLognormal, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun, method=method,
                                     **kwds)


    def update_beta(self, beta):
        self.beta = beta
        self.mu = self.exog @ beta
        
        
    def update_alpha(self, alpha):
        self.alpha = alpha
        self.sigma = np.exp(self.exog @ alpha)
    
    
    def update_expectations(self):
        kappa_0, kappa_1, kappa_2, kappa_3, mu, sigma = _gradutils(self.endog, self.exog, self.beta, self.alpha)
        self.e1 = self.mu - self.sigma * kappa_0
        sigma2 = self.sigma**2
        self.e2 = (
            sigma2 -
            sigma2 * kappa_1 +
            self.mu**2 -
            2*self.mu*self.sigma*kappa_0
        )
        

    def em(self, start_params, maxiter, use_hessian=False, step_size=1e-4, tol=1e-4):
        # Starting values
        loss = self.nloglikeobs(start_params).mean()
        self.update_beta(start_params[:11])
        self.update_alpha(start_params[11:])
        X = self.exog
        W = self.exog
        penalty_alpha = self.penalty * np.eye(11)
        WtW_plus_penalty = W.T @ W + penalty_alpha
        penalty_beta = penalty_alpha.copy()
        penalty_beta[0] = 0.
        converged = False
        
        for i in range(maxiter):
#             print(f"Iteration {i} loss: {loss}")
            loss_last = loss.copy()
            self.update_expectations()
                                    
            # Calculate beta
            X_sqrt_w = _vec_matrix_multiply(1./self.sigma, X)
            XtSiX = X_sqrt_w.T @ X_sqrt_w
            XtSiX += penalty_beta
            XtSie1 = X.T @ (self.sigma**-2 * self.e1)
            beta = np.linalg.solve(XtSiX, XtSie1)
            
            # Calculate alpha
            c = self.e2 - 2*self.e1*self.mu + self.mu**2 # NOTE: This is using the updated mu
            if use_hessian:
                grad, hessian = _em_gradutils(W, self.sigma, c, self.alpha, return_hessian=True)
                grad -= self.alpha
                hessian -= penalty_alpha
                alpha = self.alpha - np.linalg.solve(hessian, grad)
            else:
                # Backtracking line search
                grad, _ = _em_gradutils(W, self.sigma, c, self.alpha)
                d = -grad # Descent direction
                prop_increase = 0.5 # Called alpha in notes
                step_multiplier = 0.5 # Called beta in notes
                curr_step_size = step_size # Called eta in notes
                f_start = self.nloglikeobs(np.concatenate([self.beta, self.alpha])).sum()
                while True:
                    alpha = self.alpha - curr_step_size * d
                    f_stop = self.nloglikeobs(np.concatenate([self.beta, alpha])).sum()
                    required_change = prop_increase * curr_step_size * (grad @ d)
                    if f_stop - f_start <= required_change:
                        break
                    curr_step_size *= step_multiplier
            
            # Update alpha, beta simultaneously
            self.update_alpha(alpha)
            self.update_beta(beta)

            # Check convergence
            params = np.concatenate([self.beta, self.alpha])
            loss = self.nloglikeobs(params).mean()
            obj = loss_last - loss # Want this to be positive
            if abs(obj) < tol: # Not enforcing any sort of sign constraint for now
                converged = True
                break
        else:
            raise RuntimeError("Hit maxiter and failed to converge")
            
        self.params = np.concatenate([self.beta, self.alpha])
        self.iters = i
        self.loss = loss
        self.loss_last = loss_last
        self.obj = obj
        self.converged = converged