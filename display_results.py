#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = np.load('traffic-dataset.npy')[:, 1]
    nz_indices = np.where(data > 0)[0]
    y = data[nz_indices]

    gamma = np.load('output/gamma.npy')
    x = np.argmax(gamma, axis=0)                  # State

    colors = ['b', 'g', 'y', 'r']
    for i in range(3, -1, -1):
        xi = np.where(x == i)[0]
        color = colors[x[xi[0]]]
        plt.stem(nz_indices[xi], y[xi], linefmt=color+'-', markerfmt=' ')
    plt.show()

