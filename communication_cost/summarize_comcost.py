import os 
import matplotlib.pyplot as plt
import numpy as np 

LOG_PATHS = ['logs_15h31m57s.txt']
for log_path in LOG_PATHS:
    with open(log_path, 'r') as f:
        content = f.readlines()

    packet_sizes = [int(x.strip().split(',')[-1]) for x in content]
    print('{}: communication cost = {} MBs'.format(log_path, sum(packet_sizes) / float(1<<20)))