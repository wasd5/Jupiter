"""
    This script generates all possible combination of links in a fully connected network of the droplet for further network profiling procedure.
"""
__author__ = "Quynh Nguyen and Bhaskar Krishnamachari"
__copyright__ = "Copyright (c) 2019, Autonomous Networks Research Group. All rights reserved."
__license__ = "GPL"
__version__ = "2.1"

import pandas as pd
import itertools

def rand_busy_link(num):
    link_file = 'central_input/link_list.txt'
    links = pd.read_csv(link_file,header=0,delimiter = ',')
    sample_links = links.sample(num)
    print(sample_links)
    print(type(sample_links))
    output_sample_list = 'central_input/link_sample.txt'
    with open(output_sample_list,'w') as f:
        for index, row in sample_links.iterrows():
            print(row)
            f.write(row)
            f.write('\n')
            

def sim(sample_links_file):
    links = pd.read_csv(link_file,header=None,delimiter = ',')
    print(links)
    #server iperf -s -u

    #iperf -c 10.244.3.204 -u -n 1G -b 100M -d
    #iperf -c 10.244.7.175 -p 5002 -u -t 1000 -b 100M -d

if __name__ == '__main__':
    rand_busy_link(4)