#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = np.load('traffic-dataset.npy')[:, 1]
    plt.plot(data)
    plt.show()
