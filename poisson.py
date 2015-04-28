#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def compute_arrivals(data, num_zeros=10):
    arrivals = []
    count = 0

    for i in range(len(data)):
        if data[i] == 0:
            count += 1
            arrivals.append(0)
        elif data[i] != 0:
            if count >= num_zeros:
                arrivals.append(1)
            else:
                arrivals.append(0)
            count = 0

    return np.array(arrivals)

def compute_interarrival_times(arrivals):
    arrival_times = np.where(arrivals == 1)[0]
    arrival_times = np.concatenate(([-1,], arrival_times))
    interarrival_times = arrival_times[1:] - arrival_times[:-1]
    return interarrival_times

if __name__ == '__main__':
    # Load the dataset
    data = np.load('traffic-dataset.npy')

    arrivals = compute_arrivals(data[:, 1], 5)

    # Plot the dataset and the computed arrival times
    #plt.plot(data[:, 1])
    #plt.plot(5000 * arrivals)
    #plt.show()

    # Make bins out of the arrivals to find arrival counts
    arrivals = np.concatenate((arrivals, np.zeros(1)))
    N = arrivals.size
    bin_size = 50
    bins = arrivals.reshape((N//bin_size, bin_size))
    poisson_data = bins.sum(axis=1)

    # Plot the histogram of arrival counts in time-intervals of size bin_size.
    #plt.hist(poisson_data, bins=np.arange(poisson_data.max() + 1))
    #plt.show()

    interarrival_times = compute_interarrival_times(arrivals)

    # Plot histogram of inter-arrival times
    plt.hist(interarrival_times, bins=np.arange(interarrival_times.max() + 1))
    plt.show()

