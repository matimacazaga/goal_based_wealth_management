import numpy as np
import pandas as pd
from numba.typed import List
from math import ceil, sqrt, exp, log
from numba import njit, prange


@njit("(double(double))", cache=True, nogil=True)
def standard_normal_dist(x: float) -> float:
    return np.exp(-0.5*x**2)/(np.sqrt(2*np.pi))


@njit(cache=True, nogil=True, parallel=True)
def get_coefs_efficient_frontier(mean_returns_vector: np.ndarray, inv_cov_matrix: np.ndarray, cov_matrix: np.ndarray) -> tuple:

    o_vector = np.ones_like(mean_returns_vector)

    k = mean_returns_vector.T @ (inv_cov_matrix @ o_vector)

    l = mean_returns_vector.T @ (inv_cov_matrix @ mean_returns_vector)

    p = o_vector.T @ (inv_cov_matrix @ o_vector)

    denominator = l*p - k**2

    g_vector = (l*(inv_cov_matrix @ o_vector) - k *
                (inv_cov_matrix @ mean_returns_vector)) / denominator

    h_vector = (p*(inv_cov_matrix @ mean_returns_vector) -
                k*(inv_cov_matrix @ o_vector)) / denominator

    a = h_vector.T @ (cov_matrix @ h_vector)

    b = 2. * (g_vector.T @ (cov_matrix @ h_vector))

    c = g_vector.T @ (cov_matrix @ g_vector)

    return a, b, c


@njit("double(double, double, double, double)", cache=True, nogil=True)
def get_sigma(mu: float, a: float, b: float, c: float) -> float:

    return np.sqrt(a*mu**2 + b*mu + c)


@njit("(double, double, double[:], int32, double[:], double[:], int32, double)", cache=True, nogil=True, parallel=True)
def main(W_init: float, G: float, cashflows: np.ndarray, T: int,
         mu_portfolios: np.ndarray, sigma_portfolios: np.ndarray,
         i_max_init: int, h: float) -> list:

    # 2. Generar la grilla de puntos

    # a. Crear grilla vacia y agregar W_init
    grid_points = List()

    grid_points.append(np.array([W_init]))

    # b., c., d.

    sigma_max = sigma_portfolios[-1]

    mu_min, mu_max = mu_portfolios[0], mu_portfolios[-1]

    time_values = np.arange(0, T+1, 1)

    for tt in time_values[1:]:

        i_max_t = i_max_init * ceil(tt*h)
        i_array_t = np.arange(-i_max_t, i_max_t+1, 1)

        W_minus_i_max_prev = grid_points[tt-1][0]
        W_i_max_prev = grid_points[tt-1][-1]
        cashflow_prev = cashflows[tt-1]

        if W_minus_i_max_prev + cashflow_prev > 0.:  # d.1

            W_minus_i_max_t = (W_minus_i_max_prev + cashflow_prev)*exp(
                (mu_min - 0.5*sigma_max**2)*h +
                sigma_max*sqrt(h)*(-3.5)
            )

        else:  # d.2

            W_i_pos_prev = grid_points[tt -
                                       1][grid_points[tt-1] + cashflow_prev > 0.]

            assert len(W_i_pos_prev) != 0., 'Bankruptcy guaranteed'

            W_minus_i_max_t = (W_i_pos_prev[0] + cashflow_prev)*exp(
                (mu_min - 0.5*sigma_max**2)*h +
                sigma_max*sqrt(h)*(-3.5)
            )

        W_i_max_t = (W_i_max_prev + cashflow_prev) * exp(
            (mu_max - 0.5*sigma_max**2)*h +
            sigma_max * sqrt(h) * 3.5
        )

        grid_points_t = np.exp(
            ((i_array_t - (-i_max_t))/(2. * i_max_t)) *
            (log(W_i_max_t) - log(W_minus_i_max_t)) +
            log(W_minus_i_max_t)
        )

        grid_points.append(grid_points_t)

    # 3. Resolver la ec. de Bellman por backward recursion.

    value_i_t_plus_1 = np.where(grid_points[-1] >= G, 1., 0.)

    for tt in time_values[:-1][::-1]:

        transition_probabilities = np.zeros(
            shape=(grid_points[tt+1].shape[0], grid_points[tt].shape[0])
        )

        value_i_t = np.zeros_like(grid_points[tt])

        mu_i_t = np.zeros_like(grid_points[tt])

        sigma_i_t = np.zeros_like(grid_points[tt])

        for sigma, mu in zip(sigma_portfolios, mu_portfolios):

            sigma_inv = 1./sigma

            for j in prange(transition_probabilities.shape[0]):

                i_pos = np.argwhere(grid_points[tt] + cashflows[tt] > 0.)[0][0]

                for i in range(i_pos, transition_probabilities.shape[1]):

                    z = (sigma_inv) * (
                        log(grid_points[tt+1][j]/(grid_points[tt][i]+cashflows[tt])) -
                        (mu-0.5*sigma**2)
                    )

                    transition_probabilities[j, i] = standard_normal_dist(z)

            transition_probabilities = transition_probabilities / \
                transition_probabilities.sum(axis=0)

            value_i_mu = value_i_t_plus_1 @ transition_probabilities

            mask = value_i_mu > value_i_t

            value_i_t = np.where(mask, value_i_mu, value_i_t)

            mu_i_t = np.where(mask, mu, mu_i_t)

            sigma_i_t = np.where(mask, sigma, sigma_i_t)

        value_i_t_plus_1 = value_i_t

    return value_i_t, mu_i_t, sigma_i_t


if __name__ == '__main__':

    from pprint import pprint
    from time import time

    T = 10

    cov_matrix = np.array([
        [0.0017, -0.0017, -0.0021],
        [-0.0017, 0.0396, 0.03086],
        [-0.0021, 0.0309, 0.0392],
    ], order='C')

    inv_cov_matrix = np.linalg.inv(cov_matrix)

    mean_returns_vector = np.array(
        [0.0493, 0.0770, 0.0886], order='C')

    mu_min = 0.0526

    mu_max = 0.0886

    m = 15

    mu_portfolios = np.linspace(mu_min, mu_max, m)

    a, b, c = get_coefs_efficient_frontier(
        mean_returns_vector, inv_cov_matrix, cov_matrix)

    sigma_portfolios = np.array([get_sigma(mu, a, b, c)
                                 for mu in mu_portfolios])

    start = time()
    prob, mu, sigma = main(
        W_init=100.,
        G=200.,
        cashflows=np.array([0.] + [-5. for i in range(T-1)]),
        T=T,
        mu_portfolios=mu_portfolios,
        sigma_portfolios=sigma_portfolios,
        i_max_init=25,
        h=1.
    )
    end = time()

    pprint(end-start)
    print(prob, mu, sigma)
