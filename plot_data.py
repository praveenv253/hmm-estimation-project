#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = np.load('traffic-dataset.npy')[:, 1]
    plt.plot(data)
    plt.title('Internet traffic over one hour')
    plt.xlabel('Time slot')
    plt.ylabel('Number of packets (per time slot)')
    plt.show()
