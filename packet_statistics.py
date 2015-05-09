#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from misc import convert_to_binary

if __name__ == '__main__':
    data = np.load('traffic-dataset.npy')[:, 1]

    # Plot histogram of number of packets per time slot
    plt.hist(data)
    plt.show()

    # There are a lot of time slots with zero packets
    print('Fraction of slots with zero packets: %f'
            % (np.sum(data == 0) / data.size))

    # So exclude zero slots and look at the histogram of the rest of the slots
    nonzero_data = data[data != 0]
    plt.hist(nonzero_data)
    plt.show()

    # Use better binning
    bins = np.linspace(0, 5000, 200, endpoint=False)
    plt.figure(0)
    plt.hist(nonzero_data, bins=bins, log=True)
    plt.title('Histogram of number of packets in non-zero time slots')
    plt.xlabel('Number of packets')
    plt.ylabel('Frequency')
    plt.show()

    # This reveals two trends
    bins1 = np.linspace(0, 500, 20, endpoint=False)
    bins2 = np.linspace(500, 5000, 180, endpoint=False)
    plt.figure(1)
    plt.hist(nonzero_data, bins=bins1)
    plt.title('Histogram of number of packets in non-zero time slots')
    plt.xlabel('Number of packets')
    plt.ylabel('Frequency')
    plt.figure(2)
    plt.hist(nonzero_data, bins=bins2)
    plt.title('Histogram of number of packets in non-zero time slots')
    plt.xlabel('Number of packets')
    plt.ylabel('Frequency')

    # Fine-graining the bins reveals even more:
    bins1 = np.arange(501)
    plt.figure(3)
    plt.hist(nonzero_data, bins=bins1)
    plt.title('Histogram of number of packets in non-zero time slots')
    plt.xlabel('Number of packets')
    plt.ylabel('Frequency')

    plt.show()
