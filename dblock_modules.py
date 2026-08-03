import numpy as np
from scipy.stats import norm
import torch


def get_block_sigmas(
    num_layers,
    sigma_min: float = 5,
    sigma_max: float = 10,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> list[float]:
    cdf_min = norm.cdf((np.log(sigma_min) - p_mean) / p_std)
    cdf_max = norm.cdf((np.log(sigma_max) - p_mean) / p_std)
    block_sigmas = []
    for i in range(num_layers + 1):
        p = cdf_min + (cdf_max - cdf_min) * (i / num_layers)
        sigma = np.exp(p_mean + p_std * norm.ppf(p))
        block_sigmas.append(sigma)
    return block_sigmas


def get_discrete_sigmas(
    num_steps,
    sigma_min=5,
    sigma_max=10,
    rho=7.0,
    p_mean=-1.2,
    p_std=1.2,
    dblock=False,
):
    if not dblock:
        ramp = torch.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas
    else:
        log_sigma_min = np.log(sigma_min)
        log_sigma_max = np.log(sigma_max)
        cdf_min = norm.cdf((log_sigma_min - p_mean) / p_std)
        cdf_max = norm.cdf((log_sigma_max - p_mean) / p_std)
        cdf_points = np.linspace(cdf_min, cdf_max, num_steps)
        sigmas = np.exp(p_mean + p_std * norm.ppf(cdf_points))
        sigmas = torch.tensor(sigmas, dtype=torch.float32)
        sigmas = torch.flip(sigmas, dims=[0])
        return sigmas
