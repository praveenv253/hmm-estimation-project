#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = np.load('traffic-dataset.npy')[:, 1]
    nz_indices = np.where(data > 0)[0]
    y = data[nz_indices]

    gamma = np.load('output/gamma.npy')
    mu = np.load('output/mu.npy')

    x = np.argmax(gamma, axis=0)                  # State

    plt.plot(x)
    plt.show()

    colors = ['b', 'g', 'y', 'r']
    plt.figure()
    for i in range(3, -1, -1):
        xi = np.where(x == i)[0]
        color = colors[x[xi[0]]]
        plt.stem(nz_indices[xi], y[xi], linefmt=color+'-', markerfmt=' ',
                 basefmt=' ')

    plt.title('Data, coloured by HMM state\n'
              '(state defined by Poisson parameter $\mu$)')
    plt.xlabel('Time slot')
    plt.ylabel('Number of packets per time slot')
    legend_text = tuple(r'$\mu = %f$' % mu[i] for i in range(3, -1, -1))
    plt.legend(legend_text)
    plt.show()
