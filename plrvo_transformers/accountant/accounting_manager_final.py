import abc
import math
from typing import Union, Dict, Optional


from . import rdp_accounting_final as rdp_accounting

DEFAULT_ALPHAS = tuple(range(2, 256))
DEFAULT_SENSITIVITY = 1


# copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/accounting_manager.py
class AccountingManager(abc.ABC):
    def _get_sigma_with_target_epsilon(
        self,
        target_epsilon,
        target_delta,
        sample_rate,
        steps,
        threshold,
        sigma_hi_init,
        sigma_lo_init,
    ):
        """Binary search σ given ε and δ."""
        if sigma_lo_init > sigma_hi_init:
            raise ValueError("`sigma_lo` should be smaller than `sigma_hi`.")

        # Find an appropriate region for binary search.
        sigma_hi = sigma_hi_init
        sigma_lo = sigma_lo_init

        # Ensure sigma_hi isn't too small.
        while True:
            eps = self._compute_epsilon_from_sigma(sigma_hi, sample_rate, target_delta, steps)
            if eps < target_epsilon:
                break
            sigma_hi *= 2

        # Ensure sigma_lo isn't too large.
        while True:
            eps = self._compute_epsilon_from_sigma(sigma_lo, sample_rate, target_delta, steps)
            if eps > target_epsilon:
                break
            sigma_lo /= 2

        # Binary search.
        while sigma_hi - sigma_lo > threshold:
            sigma = (sigma_hi + sigma_lo) / 2
            eps = self._compute_epsilon_from_sigma(sigma, sample_rate, target_delta, steps)
            if eps < target_epsilon:
                sigma_hi = sigma
            else:
                sigma_lo = sigma

        # Conservative estimate.
        return sigma_hi

    @abc.abstractmethod
    def compute_epsilon(self, sigma, sample_rate, target_delta, steps) -> Dict:
        """Override for reporting results."""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_epsilon_from_sigma(self, sigma, sample_rate, target_delta, steps) -> float:
        """Override for binary sigma search."""
        raise NotImplementedError

    def compute_sigma(
        self,
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        epochs: Optional[Union[float, int]] = None,
        steps=None,
        threshold=1e-3,
        sigma_hi_init=4,
        sigma_lo_init=0.1,
    ) -> float:
        if steps is None:
            if epochs is None:
                raise ValueError("Epochs and steps cannot both be None.")
            steps = math.ceil(epochs / sample_rate)
        return self._get_sigma_with_target_epsilon(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=steps,
            threshold=threshold,
            sigma_hi_init=sigma_hi_init,
            sigma_lo_init=sigma_lo_init,
        )


# PLRV-O: accounting
class RDPManager(AccountingManager):
    def __init__(self, alphas=DEFAULT_ALPHAS, L2_sensitivity=DEFAULT_SENSITIVITY):
        super(RDPManager, self).__init__()
        self._alphas = alphas
        self._L2_sensitivity = L2_sensitivity

    # copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/accounting_manager.py
    def _compute_epsilon_from_sigma(self, sigma, sample_rate, target_delta, steps):
        return self.compute_epsilon(sigma, sample_rate, target_delta, steps)["eps_rdp"]

    # copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/accounting_manager.py
    def compute_epsilon(self, sigma, sample_rate, target_delta, steps) -> Dict:
        """Compute RDP as usual, but convert to (ε, δ)-DP based on the result by Canonne, Kamath, Steinke."""
        rdp = rdp_accounting.compute_rdp(q=sample_rate, noise_multiplier=sigma, steps=steps, orders=self._alphas)
        eps, alpha = rdp_accounting.get_privacy_spent(orders=self._alphas, rdp=rdp, delta=target_delta)
        return dict(eps_rdp=eps, alpha_rdp=alpha)

    # PLRV-O: accounting new
    def _compute_epsilon_from_distribution(self, params, sample_rate, target_delta, steps):
        return self.compute_epsilon_p(params, sample_rate, target_delta, steps)["epsilon"]
    
    # PLRV-O: accounting new
    def compute_epsilon_p(self, params, sample_rate, target_delta, steps) -> Dict:
        """convert to (ε, δ)-DP based on the result by Canonne, Kamath, Steinke."""
        rdp = rdp_accounting.compute_rdp_p(params=params, q=sample_rate, steps=steps, orders=self._alphas, L2_sensitivity=self._L2_sensitivity)
        epsilon, alpha_rdp = rdp_accounting.get_privacy_spent(orders=self._alphas, rdp=rdp, delta=target_delta)
        return dict(epsilon=epsilon, alpha_rdp=alpha_rdp)

# copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/accounting_manager.py
class GLWManager(AccountingManager):
    def __init__(self, eps_error=0.05):
        super(GLWManager, self).__init__()
        self._eps_error = eps_error

    def _compute_epsilon_from_sigma(self, sigma, sample_rate, target_delta, steps):
        return self.compute_epsilon(sigma, sample_rate, target_delta, steps)["eps_upper"]  # Be conservative.

    def compute_epsilon(self, sigma, sample_rate, target_delta, steps) -> Dict:
        if steps == 0:
            return dict(eps_low=None, eps_estimate=None, eps_upper=None)

        from prv_accountant import Accountant
        accountant = Accountant(
            noise_multiplier=sigma,
            sampling_probability=sample_rate,
            delta=target_delta,
            eps_error=self._eps_error,
            max_compositions=steps
        )
        eps_low, eps_estimate, eps_upper = accountant.compute_epsilon(num_compositions=steps)
        return dict(eps_low=eps_low, eps_estimate=eps_estimate, eps_upper=eps_upper)
