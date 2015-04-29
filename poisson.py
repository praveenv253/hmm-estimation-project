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
    arrival_times = np.concatenate(([0,], arrival_times))
    interarrival_times = arrival_times[1:] - arrival_times[:-1]
    return interarrival_times

def mle_exponential_fit(interarrival_times):
    lamda = 1 / np.mean(interarrival_times)
    return lamda

def mvub_exponential_fit(interarrival_times):
    n = interarrival_times.size
    lamda = (n - 1) / (n * np.mean(interarrival_times))
    return lamda

def mle_pareto_fit(interarrival_times):
    n = interarrival_times.size
    k = np.min(interarrival_times)
    alpha = n / np.sum(np.log(interarrival_times / k))
    return (k, alpha)

if __name__ == '__main__':
    # Load the dataset
    data = np.load('traffic-dataset.npy')

    num_zeros = 1
    arrivals = compute_arrivals(data[:, 1], num_zeros)

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
    lamda = mle_exponential_fit(interarrival_times)
    (k, alpha) = mle_pareto_fit(interarrival_times)

    # Plot histogram of inter-arrival times
    histbins = np.arange(interarrival_times.max() + 1)
    plt.hist(interarrival_times, bins=histbins, normed=True)

    # Plot the exponential fit
    x = (histbins[1:] + histbins[:-1]) / 2
    plt.plot(x, lamda * np.exp(- lamda * x), 'r-', linewidth=2)

    # Plot the pareto fit
    x1 = x[num_zeros:]
    plt.plot(x1, alpha * k**alpha / x1**(alpha+1), 'g-', linewidth=2)

    plt.title('Histogram of inter-arrival times, with exp and pareto fits\n'
              '(# zeros between bursts = %d)' % num_zeros)
    plt.xlabel('Inter arrival time')
    plt.ylabel('Probability')
    plt.legend(('Exponential fit', 'Pareto fit'))

    plt.show()
