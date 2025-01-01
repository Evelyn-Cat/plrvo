import math, warnings
import numpy as np
from scipy import special
from typing import List, Sequence, Union, Dict, Optional
from math import log, exp, floor, inf


########################
# LOG-SPACE ARITHMETIC #
########################


# copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/rdp_accounting.py
def _log_add(logx: float, logy: float) -> float:
    r"""Adds two numbers in the log space.

    Args:
        logx: First term in log space.
        logy: Second term in log space.

    Returns:
        Sum of numbers in log space.
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


# copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/rdp_accounting.py
def _log_sub(logx: float, logy: float) -> float:
    r"""Subtracts two numbers in the log space.

    Args:
        logx: First term in log space. Expected to be greater than the second term.
        logy: First term in log space. Expected to be less than the first term.

    Returns:
        Difference of numbers in log space.

    Raises:
        ValueError
            If the result is negative.
    """
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


# copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/rdp_accounting.py
def _compute_log_a_for_int_alpha(q: float, sigma: float, alpha: int) -> float:
    r"""Computes :math:`log(A_\alpha)` for integer ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """

    # Initialize with 0 in the log space.
    log_a = -np.inf

    for i in range(alpha + 1):
        log_coef_i = (
            math.log(special.binom(alpha, i))
            + i * math.log(q)
            + (alpha - i) * math.log(1 - q)
        )

        s = log_coef_i + (i * i - i) / (2 * (sigma ** 2))
        log_a = _log_add(log_a, s)

    return float(log_a)


# copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/rdp_accounting.py
def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for fractional ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1)


# copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/rdp_accounting.py
def _compute_log_a(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for any positive finite ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf
        for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in the paper mentioned above.
    """
    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)


# copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/rdp_accounting.py
def _log_erfc(x: float) -> float:
    r"""Computes :math:`log(erfc(x))` with high accuracy for large ``x``.

    Helper function used in computation of :math:`log(A_\alpha)`
    for a fractional alpha.

    Args:
        x: The input to the function

    Returns:
        :math:`log(erfc(x))`
    """
    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)


# copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/rdp_accounting.py
def _compute_rdp(q: float, sigma: float, alpha: float) -> float:
    r"""Computes RDP of the Sampled Gaussian Mechanism at order ``alpha``.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        RDP at order ``alpha``; can be np.inf.
    """
    if q == 0:
        return 0

    # no privacy
    if sigma == 0:
        return np.inf

    if q == 1.0:
        return alpha / (2 * sigma ** 2)

    if np.isinf(alpha):
        return np.inf

    return _compute_log_a(q, sigma, alpha) / (alpha - 1)


# copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/rdp_accounting.py
def compute_rdp(
    q: float, noise_multiplier: float, steps: int, orders: Union[Sequence[float], float]
) -> Union[List[float], float]:
    r"""Computes Renyi Differential Privacy (RDP) guarantees of the
    Sampled Gaussian Mechanism (SGM) iterated ``steps`` times.

    Args:
        q: Sampling rate of SGM.
        noise_multiplier: The ratio of the standard deviation of the
            additive Gaussian noise to the L2-sensitivity of the function
            to which it is added. Note that this is same as the standard
            deviation of the additive Gaussian noise when the L2-sensitivity
            of the function is 1.
        steps: The number of iterations of the mechanism.
        orders: An array (or a scalar) of RDP orders.

    Returns:
        The RDP guarantees at all orders; can be ``np.inf``.
    """
    if isinstance(orders, float):
        rdp = _compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array([_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps


# PLRV-O: MGF of distributions
def Mu(
    t: float,
    params: Dict
) -> Optional[float]:
    assert params != None and params["distributions"] != None

    MGF = 1
    distributions_used = []
    if "Gamma" in params["distributions"]:
        try:
            M_G = ((1-params['a_G']*t*params['G_theta'])**(-params['G_k']))
            MGF = MGF * M_G
            distributions_used.append("Gamma")
        except:
            pass
    
    if "Exponential" in params["distributions"]:
        try:
            M_E = (params['E_lambda']/(params['E_lambda']-params['a_E']*t))
            MGF = MGF * M_E
            distributions_used.append("Exponential")
        except:
            pass
    
    if distributions_used != params["distributions"]:
        # not in search space
        return 0
    else:
        return MGF


# PLRV-O: TODO Eq. 6 for each \alpha_{M_q}(\lambda)
def _compute_plrvo_for_int_alpha(q, params, alpha, L2_sensitivity) -> float:
    summation = 0
    for eta in range(alpha + 1):
        # TODO avoid overflow
        log_factorials = np.zeros(alpha + 1)
        log_factorials[1:] = np.cumsum(np.log(np.arange(1, alpha + 1)))
        log_binom_coeff = (log_factorials[alpha] - log_factorials[eta] - log_factorials[alpha - eta])
        binom_coeff = np.exp(log_binom_coeff)
        # binom_coeff = math.comb(alpha, eta)

        weight = ((1 - q) ** (alpha - eta)) * (q ** eta)
        
        t = L2_sensitivity * eta
        if t<=1/params["G_theta"]:
            summation = summation + binom_coeff * weight * Mu(t=t, params=params)
        
        # if binom_coeff * weight * Mu(t=t, params=params)<0: TODO
        #     print(alpha, eta, binom_coeff * weight * Mu(t=t, params=params))
    
    if summation == 0:
        return np.inf
    
    if math.log(summation) < 0:
        # TODO print("math.log(summation) < 0: when ", alpha, eta)
        return np.inf
    else:
        return math.log(summation)



# PLRV-O: TODO Eq. 6
def _compute_rdp_order_p(
    q: float, params: Dict, alpha: int, L2_sensitivity: float
) -> float:
    r"""Computes RDP of the Sampled PLRV-O Mechanism at order ``alpha``.

    Args:
        q: Sampling rate of SPM.
        params: The PLRV-O noise parameters.
        alpha: The order at which RDP is computed.
        L2_sensitivity: The clipping threshold.

    Returns:
        RDP at order ``alpha``; can be np.inf.
    """
    if q == 0:
        return 0

    if np.isinf(alpha):
        return np.inf

    return _compute_plrvo_for_int_alpha(q, params, alpha, L2_sensitivity)


# PLRV-O: TODO Eq. 6
def compute_rdp_p(
    q: float, params: Dict, steps: int, orders: Union[Sequence[int], int], L2_sensitivity: float
) -> Union[List[float], float]:
    r"""Computes Renyi Differential Privacy (RDP) guarantees of the
    Sampled PLRV-O Mechanism (SPM) iterated ``steps`` times.

    Args:
        q: Sampling rate of SPM.
        params: PLRV-O parameters including distribution names, 
          distribution parameters and clipping threshold.
        steps: The number of iterations of the mechanism.
        orders: An array (or a scalar) of RDP orders.
        L2_sensitivity: The clipping threshold.

    Returns:
        The RDP guarantees at all orders; can be ``np.inf``.
    """
    rdp = np.array([_compute_rdp_order_p(q, params, order, L2_sensitivity) for order in orders])

    return rdp * steps


# copied from https://github.com/lxuechen/private-transformers/blob/main/private_transformers/accounting/rdp_accounting.py
# Based on
#   https://github.com/tensorflow/privacy/blob/5f07198b66b3617b22609db983926e3ba97cd905/tensorflow_privacy/privacy/analysis/rdp_accountant.py#L237
def get_privacy_spent(orders, rdp, delta):
    """Compute epsilon given a list of RDP values and target delta.
    Args:
        orders: An array (or a scalar) of orders.
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of (eps, optimal_order).
    Raises:
        ValueError: If input is malformed.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if delta <= 0:
        raise ValueError("Privacy failure probability bound delta must be >0.")
    if len(orders_vec) != len(rdp_vec):
        raise ValueError("Input lists must have the same length.")

    # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
    #   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )

    # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
    # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
    eps_vec = []
    for (a, r) in zip(orders_vec, rdp_vec):
        if a < 1:
            raise ValueError("Renyi divergence order must be >=1.")
        if r < 0:
            raise ValueError("Renyi divergence must be >=0.")

        if delta ** 2 + math.expm1(-r) >= 0:
            # In this case, we can simply bound via KL divergence:
            # delta <= sqrt(1-exp(-KL)).
            eps = 0  # No need to try further computation if we have eps = 0.
        elif a > 1.01:
            # This bound is not numerically stable as alpha->1.
            # Thus we have a min value of alpha.
            # The bound is also not useful for small alpha, so doesn't matter.
            eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
        else:
            # In this case we can't do anything. E.g., asking for delta = 0.
            eps = np.inf
        eps_vec.append(eps)
    
    idx_opt = np.argmin(eps_vec)
    return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


def cmp_lhs(params, clip, eta, distributions):
    params["distributions"] = distributions
    if params["distributions"] == ["Gamma"]:
        G_k, G_theta = params["G_k"], params["G_theta"]
        exponent = 0.5 * (eta**2-eta) * clip**2 * (G_k-1)**2 * G_theta**2
        if exponent > 709:
            # TODO avoid overflow
            return None
        else:
            return np.exp(exponent)
    else:
        # TODO
        exit(0)


def cmp_rhs(params, clip, eta, distributions):
    params["distributions"] = distributions
    if params['distributions'] == ["Gamma"]:
        G_k, G_theta = params["G_k"], params["G_theta"]
        x = eta * clip
        y = (eta - 1) * clip
        if 1 - y * G_theta <= 0:
            return None
    
        with warnings.catch_warnings(record=True) as w:
            # avoid overflow
            warnings.simplefilter("always")
            M1 = 0.5 * (1 - y * G_theta)**(-G_k)
            if len(w) > 0:
                return None
        
        M2 = (1 - (0.5 * (1+x*G_theta)**(-G_k)) - (0.5 * (1+y*G_theta)**(-G_k))) / (2*eta-1)
        M3 = 0.5 * (1 + x*G_theta)**(-G_k)
        return M1 + M2 + M3
    else:
        # TODO
        exit(0)


def _compute_plrvo_for_int_alpha2(q, params, alpha, L2_sensitivity) -> float:
    C = L2_sensitivity
    clip = L2_sensitivity
    theta = params["G_theta"]
    k = params["G_k"]
    
    eps_check = inf
    best_lambda = 1
    bound = min(200, floor(1 / (clip * theta)))

    for lambda_ in range(1, bound + 1):
        lmbda1 = lambda_ + 1

        # Compute log factorials
        log_factorials = [0] + list(np.cumsum(np.log(np.arange(1, lmbda1 + 1))))

        # Initialize the summation
        M = 0

        # Loop through all k values
        for kk in range(0, lmbda1 + 1):
            if kk == 0:
                rhs = 1
            else:
                x = kk * C
                y = (kk - 1) * C

                A = 0.5 * (1 - y * theta) ** (-k)
                B = (1 - (0.5 * (1 + x * theta) ** (-k)) - (0.5 * (1 + y * theta) ** (-k))) / ((2 * kk) - 1)
                Cp = 0.5 * (1 + x * theta) ** (-k)

                rhs = A + B + Cp

            log_binom_coeff = (
                log_factorials[lmbda1] - log_factorials[kk] - log_factorials[lmbda1 - kk]
            )
            binom_coeff = exp(log_binom_coeff)  # Convert back from log scale

            # Compute the term
            term = binom_coeff * ((1 - q) ** (lmbda1 - kk)) * (q ** kk) * rhs

            M += term

        M = log(M) / lambda_

    return M



def _compute_rdp_order_p2(
    q: float, params: Dict, alpha: int, L2_sensitivity: float
) -> float:
    r"""Computes RDP of the Sampled PLRV-O Mechanism at order ``alpha``.

    Args:
        q: Sampling rate of SPM.
        params: The PLRV-O noise parameters.
        alpha: The order at which RDP is computed.
        L2_sensitivity: The clipping threshold.

    Returns:
        RDP at order ``alpha``; can be np.inf.
    """
    if q == 0:
        return 0

    if np.isinf(alpha):
        return np.inf

    return _compute_plrvo_for_int_alpha2(q, params, alpha, L2_sensitivity)


def compute_rdp_p2(
    q: float, params: Dict, steps: int, orders: Union[Sequence[int], int], L2_sensitivity: float
) -> Union[List[float], float]:
    r"""Computes Renyi Differential Privacy (RDP) guarantees of the
    Sampled PLRV-O Mechanism (SPM) iterated ``steps`` times.

    Args:
        q: Sampling rate of SPM.
        params: PLRV-O parameters including distribution names, 
          distribution parameters and clipping threshold.
        steps: The number of iterations of the mechanism.
        orders: An array (or a scalar) of RDP orders.
        L2_sensitivity: The clipping threshold.

    Returns:
        The RDP guarantees at all orders; can be ``np.inf``.
    """
    rdp = np.array([_compute_rdp_order_p2(q, params, order, L2_sensitivity) for order in orders])

    return rdp * steps

def _compute_plrvo_for_int_alpha3(q, params, alpha, L2_sensitivity) -> float:
    C = L2_sensitivity
    clip = L2_sensitivity
    
    theta = params["G_theta"]
    k = params["G_k"]
    
    eps_check = inf
    best_lambda = 1
    bound = min(200, floor(1 / (clip * theta)))

    for lambda_ in range(1, bound + 1):
        lmbda1 = lambda_ + 1

        # Compute log factorials
        log_factorials = [0] + list(np.cumsum(np.log(np.arange(1, lmbda1 + 1))))

        # Initialize the summation
        M = 0

        # Loop through all k values
        for kk in range(0, lmbda1 + 1):
            if kk == 0:
                rhs = 1
            else:
                x = -kk * C
                y = (kk - 1) * C

                A = 0.5 * (1 - y * theta) ** (-k)
                B = ((1 - x * theta) ** (-k) +(1 - y * theta) ** (-k)) / (2*((2 * kk) - 1))
                Cp = 0.5 * (1 - x * theta) ** (-k)

                rhs = A + B + Cp

            log_binom_coeff = (
                log_factorials[lmbda1] - log_factorials[kk] - log_factorials[lmbda1 - kk]
            )
            binom_coeff = exp(log_binom_coeff)  # Convert back from log scale

            # Compute the term
            term = binom_coeff * ((1 - q) ** (lmbda1 - kk)) * (q ** kk) * rhs

            M += term

        M = log(M) / lambda_
    
    return M



def _compute_rdp_order_p3(
    q: float, params: Dict, alpha: int, L2_sensitivity: float
) -> float:
    r"""Computes RDP of the Sampled PLRV-O Mechanism at order ``alpha``.

    Args:
        q: Sampling rate of SPM.
        params: The PLRV-O noise parameters.
        alpha: The order at which RDP is computed.
        L2_sensitivity: The clipping threshold.

    Returns:
        RDP at order ``alpha``; can be np.inf.
    """
    if q == 0:
        return 0

    if np.isinf(alpha):
        return np.inf

    return _compute_plrvo_for_int_alpha3(q, params, alpha, L2_sensitivity)


def compute_rdp_p3(
    q: float, params: Dict, steps: int, orders: Union[Sequence[int], int], L2_sensitivity: float
) -> Union[List[float], float]:
    r"""Computes Renyi Differential Privacy (RDP) guarantees of the
    Sampled PLRV-O Mechanism (SPM) iterated ``steps`` times.

    Args:
        q: Sampling rate of SPM.
        params: PLRV-O parameters including distribution names, 
          distribution parameters and clipping threshold.
        steps: The number of iterations of the mechanism.
        orders: An array (or a scalar) of RDP orders.
        L2_sensitivity: The clipping threshold.

    Returns:
        The RDP guarantees at all orders; can be ``np.inf``.
    """
    rdp = np.array([_compute_rdp_order_p3(q, params, order, L2_sensitivity) for order in orders])
    

    return rdp * steps
