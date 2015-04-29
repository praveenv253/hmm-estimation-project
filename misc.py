#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress

def autocorr(data, n):
    zero_mean_data = data - np.mean(data)
    corrs = []
    corrs.append(1)
    normalizer = np.sum(zero_mean_data ** 2)
    for i in range(1, n):
        corr = np.sum(zero_mean_data[i:] * zero_mean_data[:-i]) / normalizer
        corrs.append(corr)
    return np.array(corrs)

def convert_to_binary(data):
    binary_data = data.copy()
    binary_data[data != 0] = 1
    return binary_data

def difference(data, d):
    return data[d:] - data[:-d]

def aggregate(data, m):
    n = data.size // m
    ret = data[:n*m].reshape((n, m))
    return ret.mean(axis=1)

def variance_time(data, n):
    variances = []
    variances.append(np.var(data))
    for i in range(1, n):
        variances.append(np.var(aggregate(data, i+1)))
    return np.array(variances)

if __name__ == '__main__':
    data = np.load('traffic-dataset.npy')[:, 1]

    #binary_data = convert_to_binary(data)

    #diff = difference(binary_data, 1)
    #plt.plot(binary_data[:100])
    #plt.plot(convert_to_binary(diff[:100]))
    #plt.show()
    #diff = difference(convert_to_binary(diff), 100)
    #corrs = autocorr(convert_to_binary(diff), 500)
    #corrs = autocorr(data, 500)

    # Self-similar processes have a slow-decay variance-time plot
    variances = variance_time(data, 30)
    variances /= variances[0]
    plt.loglog(np.arange(variances.size) + 1, variances)
    plt.show()

    # Compute Hurst parameter
    m = np.arange(variances.size) + 1
    x = np.log(m)
    y = np.log(variances)
    slope = linregress(x, y)[0]                         # -0.386454
    print('Slope of variance-time graph: %f' % slope)
    hurst_parameter = 1 + slope / 2                     # 0.806773
    print('Hurst parameter: %f' % hurst_parameter)
    # A Hurst parameter close to 1 and far from 0.5 indicates a more
    # self-similar process

