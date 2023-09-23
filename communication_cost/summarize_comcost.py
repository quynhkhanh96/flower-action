import sys 
from datetime import datetime

log_fpath = sys.argv[1]
with open(log_fpath, 'r') as f:
    logs = f.readlines()

logs = [int(log.strip().split(' ')[-2]) for log in logs if 'sending' in log]
exp_name = log_fpath.split('/')[-1].split('.')[0]
print('Exp {} has communication cost of {} MBs.'.format(exp_name, sum(logs) / float(1<<20)))

start_time = logs[0].split(' - ')[0]
end_time = logs[-1].split(' - ')[0]

st = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
et = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
tot = et - st 
print('Exp {} took {} hours.'.format(exp_name, tot.total_seconds() / 3600))
