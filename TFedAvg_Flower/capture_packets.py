from scapy.all import *
import os 
from datetime import datetime

# callback function - called for every packet
LOG_PATH = 'logs.txt'

def traffic_monitor_callback(pkt):
    pkt_size = len(pkt)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    pkt_src = pkt[0][1].src 
    pkt_dst = pkt[0][1].dst 
    msg = f'[{current_time}]: {pkt_size}'
    print(msg)
    with open(LOG_PATH, 'a') as f:
        f.write(f'{current_time},{pkt_size}\n')

if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

# capture traffic
sniff(filter="port 8085", iface='lo', prn=traffic_monitor_callback)