#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.special as spl
import numpy.linalg as linalg

FLOAT = np.float128

def compute_b(y, mu):
    return np.exp(y * np.log(mu) - mu  - spl.gammaln(y + 1))

def compute_alpha(y, q, A, mu):
    N = A.shape[0]
    T = y.shape[0]
    alpha = np.zeros((N, T), dtype=FLOAT)
    alpha[:, 0] = q * compute_b(y[0], mu)
    alpha[:, 0] /= np.sum(alpha[:, 0])
    for t in range(T - 1):
        b_ytp1 = compute_b(y[t+1], mu)
        alpha[:, t+1] = b_ytp1 * np.dot(alpha[:, t], A)
        normalizer = np.sum(alpha[:, t+1])
        if normalizer > 0.0:
            alpha[:, t+1] /= np.sum(alpha[:, t+1])
        else:
            raise ValueError('Alpha will produce a nan')
    return alpha

def compute_beta(y, q, A, mu):
    N = A.shape[0]
    T = y.shape[0]
    beta = np.zeros((N, T), dtype=FLOAT)
    beta[:, -1] = 1. / N
    for t in range(T-2, -1, -1):
        b_ytp1 = compute_b(y[t+1], mu)
        beta[:, t] = np.dot(A, beta[:, t+1] * b_ytp1)
        normalizer = np.sum(beta[:, t+1])
        if normalizer > 0.0:
            beta[:, t] /= np.sum(beta[:, t])
        else:
            raise ValueError('Beta will produce a nan')
    return beta

def compute_gamma(alpha, beta):
    gamma = alpha * beta
    gamma /= np.sum(gamma, axis=0)
    return gamma

def compute_xi(y, A, mu, alpha, beta):
    b = compute_b(y, mu[:, None])
    xi = np.einsum('it,ij,jt,jt->ijt', alpha[:, :-1], A, b[:, 1:], beta[:, 1:])
    xi /= np.sum(alpha * beta, axis=0)[:-1]
    return xi

def update(y, q, A, mu, alpha, beta, gamma, xi):
    q_new = gamma[:, 1]
    A_new = xi.sum(axis=2) / gamma[:, :-1].sum(axis=1)
    mu_new = np.sum(gamma * y, axis=1) / gamma.sum(axis=1)

    # Normalize everything for consistency
    q_new /= q_new.sum()
    A_new /= A_new.sum(axis=1)[:, None]
    return (q_new, A_new, mu_new)


if __name__ == '__main__':

    data = np.load('traffic-dataset.npy')[:, 1]
    y = data[data > 0]

    num_states = 4
    A = np.ones((num_states, num_states)) / num_states
    q = np.ones(num_states) / num_states
    mu = np.array([1, 100, 1000, 10000], dtype=FLOAT)   # Need to change these

    # Number of iterations of EM
    K = 10
    for k in range(K):
        print('------------ Iteration %d of EM ------------' % (k+1))
        # First perform the forward & backward procedures
        alpha = compute_alpha(y, q, A, mu)
        beta = compute_beta(y, q, A, mu)
        print('alpha')
        print(alpha)
        print('beta')
        print(beta)

        # Perform state estimation
        gamma = compute_gamma(alpha, beta)
        print('gamma')
        print(gamma)

        xi = compute_xi(y, A, mu, alpha, beta)
        print('xi')
        print(xi)

        (q_new, A_new, mu_new) = update(y, q, A, mu, alpha, beta, gamma, xi)
        q = q_new
        A = A_new
        mu = mu_new
        print('q')
        print(q)
        print('A')
        print(A)
        print('mu')
        print(mu)
        print()
