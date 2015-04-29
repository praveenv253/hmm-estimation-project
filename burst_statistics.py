#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from poisson import compute_arrivals, compute_interarrival_times

if __name__ == '__main__':
    data = np.load('traffic-dataset.npy')[:, 1]

    num_zeros = 5
    arrivals = compute_arrivals(data, num_zeros)
    interarrival_times = compute_interarrival_times(arrivals)
    arrival_indices = np.cumsum(interarrival_times)
    slices = np.split(np.arange(data.size), arrival_indices)
    burst_sizes = []
    for s in slices:
        burst_sizes.append(np.sum(data[s]))

    data = np.array(burst_sizes)
    savedata = np.hstack((arrival_indices[:, None], data[1:, None]))
    np.save('bursts_%d' % num_zeros, savedata)

    # Plot histogram of number of packets per time slot
    plt.hist(data, bins=np.linspace(0, 10000, 500, endpoint=False))
    plt.show()
