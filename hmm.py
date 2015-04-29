#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import scipy.special as spl
import numpy.linalg as linalg

def f(g, w, x):
    return (x / w) ** (g - 1) * np.exp(- x / w) / (w * spl.gamma(g))

def fy(y, i, params):
    (gt, wt, gp, wp) = params
    return f(gt[i], wt[i], y[0]) * f(gp[i], wp[i], y[1])

def compute_alpha(A, y, params):
    num_states = A.shape[0]
    Lk = y.shape[0]
    alpha = np.zeros((Lk, num_states))
    alpha[0, 0] = 1
    for n in range(1, Lk):
        for j in range(num_states):
            alpha[n, j] = np.sum(alpha[n-1, :] * A[:, j]) * fy(y[n], j, params)
    return alpha

def compute_beta(A, y, params):
    num_states = A.shape[0]
    Lk = y.shape[0]
    beta= np.zeros((Lk, num_states))
    beta[-1, :] = 1
    for n in range(Lk-2, 0, -1):
        for i in range(num_states):
            for j in range(num_states):
                beta[n, i] += A[i, j] * fy(y[n+1], j, params) * beta[n+1, j]
    return beta

def update(A, y, alpha, beta, params):
    num_states = A.shape[0]
    Lk = y.shape[0]
    gtwt = np.zeros(num_states)
    gtwt2 = np.zeros(num_states)
    gpwp = np.zeros(num_states)
    gpwp2 = np.zeros(num_states)
    A_new = A.copy()
    (gt, wt, gp, wp) = params
    for i in range(num_states):
        for j in range(num_states):
            fyn = []
            for n in range(Lk):
                fyn.append( fy(y[n], j, params) )
            fyn = np.array(fyn)
            A_new[i, j] = np.sum(alpha[:-1, i] * A[i, j] * fyn[1:] * beta[1:, j]) / np.sum(alpha[:-1, i] * beta[:-1, i])
        gtwt[i] = np.sum(alpha[:, i] * beta[:, i] * y[:, 0]) / np.sum(alpha[:-1, i] * beta[:-1, i])
        gpwp[i] = np.sum(alpha[:, i] * beta[:, i] * y[:, 1]) / np.sum(alpha[:-1, i] * beta[:-1, i])
        gtwt2[i] = np.sum(alpha[:, i] * beta[:, i] * (y[:, 0] - gtwt[i])**2) / np.sum(alpha[:-1, i] * beta[:-1, i])
        gtwt2[i] = np.sum(alpha[:, i] * beta[:, i] * (y[:, 1] - gpwp[i])**2) / np.sum(alpha[:-1, i] * beta[:-1, i])

    # Get back parameters
    i1 = (abs(gtwt) > 1e-5)
    i2 = (abs(gpwp) > 1e-5)
    wt[i1] = gtwt2[i1] / gtwt[i1]
    wp[i2] = gpwp2[i2] / gpwp[i2]
    i3 = (abs(wt) > 1e-5)
    i4 = (abs(wp) > 1e-5)
    gt[i3] = gtwt[i3] / wt[i3]
    gp[i4] = gpwp[i4] / wp[i4]

    new_params = (gt, wt, gp, wp)
    return (A_new, new_params)


if __name__ == '__main__':

    y = np.load('bursts_5.npy')

    num_states = 3
    A = np.ones((num_states, num_states)) / num_states
    q = np.ones(num_states) / num_states

    dmin = y[:, 0].min()
    dmax = y[:, 0].max()
    bmin = y[:, 0].min()
    bmax = y[:, 0].max()

    #mut = np.arange(num_states) * (dmax - dmin) / (num_states+1) + 1
    #mup = np.arange(num_states) * (bmax - bmin) / (num_states+1) + 1
    #sigmat = (dmax - dmin) / (5 * (num_states+1))
    #sigmap = (bmax - bmin) / (5 * (num_states+1))

    mut = np.array([10, 30, 100])
    mup = np.array([200, 800, 2000])
    sigmat = np.array([100, 900, 1e4])
    sigmap = np.array([200, 800, 2000])

    # Initialize params
    wt = sigmat ** 2 / mut
    wp = sigmap ** 2 / mup
    gt = mut / wt
    gp = mup / wp

    params = (gt, wt, gp, wp)

    for k in range(1):
        alpha = compute_alpha(A, y, params)
        beta = compute_beta(A, y, params)
        print(alpha)
        print(beta)
        (A_new, new_params) = update(A, y, alpha, beta, params)
        A = A_new
        params = new_params
        print(k)
        print(params)
        print(A)
        print('-------')

    (eigvals, eigvecs) = linalg.eig(A)
    index = np.argmin(abs(eigvals - 1))    # Find the eigenvalue closest to 1
    q = eigvecs[:, index]

    print(q)

