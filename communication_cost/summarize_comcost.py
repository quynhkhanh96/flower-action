import sys 
import matplotlib.pyplot as plt
import numpy as np 

log_path = sys.argv[1]
with open(log_path, 'r') as f:
    content = f.readlines()

packet_sizes = [int(x.strip().split(',')[-1]) for x in content]
exp_name = log_path.split('/')[-1].split('.')[0]
print('Exp at {}: communication cost = {} MBs'.format(exp_name, sum(packet_sizes) / float(1<<20)))