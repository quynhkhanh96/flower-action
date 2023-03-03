import sys 

log_fpath = sys.argv[1]
with open(log_fpath, 'r') as f:
    logs = f.readlines()

logs = [int(log.strip().split(' ')[-2]) for log in logs if 'sending' in log]
exp_name = log_fpath.split('/')[-1].split('.')[0]
print('Exp {} has communication cost of {} MBs'.format(exp_name, sum(logs) / float(1<<20)))