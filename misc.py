#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def autocorr(data, n):
    corrs = []
    corrs.append(1)
    normalizer = np.sum(data ** 2)
    for i in range(1, n):
        corr = np.sum(data[i:] * data[:-i]) / normalizer
        corrs.append(corr)
    return np.array(corrs)

def convert_to_binary(data):
    binary_data = data.copy()
    binary_data[data != 0] = 1
    return binary_data

def difference(data, d):
    return data[d:] - data[:-d]

if __name__ == '__main__':
    data = np.load('traffic-dataset.npy')[:, 1]

    binary_data = convert_to_binary(data)

    diff = difference(binary_data, 1)
    plt.plot(binary_data[:100])
    plt.plot(convert_to_binary(diff[:100]))
    plt.show()
    diff = difference(convert_to_binary(diff), 100)
    corrs = autocorr(convert_to_binary(diff), 500)

    plt.plot(corrs)
    plt.show()

