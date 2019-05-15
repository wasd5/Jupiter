__author__ = "Quynh Nguyen and Bhaskar Krishnamachari"
__copyright__ = "Copyright (c) 2019, Autonomous Networks Research Group. All rights reserved."
__license__ = "GPL"
__version__ = "2.1"

from tornado import gen
from functools import partial
from bokeh.models import Button, NumeralTickFormatter
from bokeh.palettes import RdYlBu3
from bokeh.plotting import *
from bokeh.core.properties import value
from bokeh.models import ColumnDataSource, Label, LabelSet, HoverTool,Circle
from bokeh.models import TapTool, BoxSelectTool, GraphRenderer, StaticLayoutProvider
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.io import output_file, show
from bokeh.layouts import row,column
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.palettes import brewer
from bokeh.transform import linear_cmap
from bokeh.layouts import widgetbox,layout
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models.widgets import MultiSelect, Select, Div
from bokeh.models.graphs import from_networkx,NodesOnly
from bokeh.models import Label
from bokeh.models import Arrow, OpenHead
from datetime import date
from datetime import datetime
from random import randint
import paho.mqtt.client as mqtt
import time
import numpy as np 
from itertools import cycle, islice
import random
import pandas as pd
import datetime
import collections
from pytz import timezone
import configparser
import itertools
from itertools import product, combinations, chain
import networkx as nx




class mq():


    def __init__(self,outfname,subs,server,port,timeout,looptimeout):
        self.OUTFNAME = outfname
        self.subs = subs
        self.server = server
        self.port = port
        self.timeout = timeout
        self.looptimeout = looptimeout
        self.outf = open(OUTFNAME,'a')
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.server, self.port, self.timeout)

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self,client, userdata, flags, rc):
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        subres = client.subscribe(self.subs,qos=1)
        print("Connected with result code "+str(rc))


    # The callback for when a PUBLISH message is received from the server.
    def on_message(self,client, userdata, msg):
        start_messages = ['localpro starts', 'aggregate0 starts', 'aggregate1 starts', 'aggregate2 starts',
        'simpledetector0 starts', 'simpledetector1 starts', 'simpledetector2 starts', 'astutedetector0 starts',
        'astutedetector1 starts', 'astutedetector2 starts', 'fusioncenter0 starts', 'fusioncenter1 starts', 
        'fusioncenter2 starts', 'globalfusion starts']

        end_messages = ['localpro ends', 'aggregate0 ends', 'aggregate1 ends', 'aggregate2 ends',
        'simpledetector0 ends', 'simpledetector1 ends', 'simpledetector2 ends', 'astutedetector0 ends',
        'astutedetector1 ends', 'astutedetector2 ends', 'fusioncenter0 ends', 'fusioncenter1 ends', 
        'fusioncenter2 ends', 'globalfusion ends']


        top_dag=[5.15, 4.15, 4.15, 4.15, 3.15, 3.15, 3.15, 3.15, 3.15, 3.15, 2.15,2.15,2.15,1.15]
        bottom_dag=[4.85, 3.85,3.85,3.85, 2.85,2.85,2.85,2.85,2.85,2.85, 1.85,1.85,1.85, 0.85]
        left_dag= [3.3,1.3,3.3,5.3, 0.8, 1.8, 2.8, 3.8, 4.8,5.8, 1.3,3.3,5.3,3.3]
        right_dag=[3.7,1.7,3.7,5.7, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2,1.7,3.7,5.7,3.7]


        message = msg.payload.decode()
        global start_time
        global finish_time
        global total_time

        print('--------------')
        print(message)
        print('--------------')
    

        if message.startswith('mapping'):
            print('---- Receive task mapping')
            doc.add_next_tick_callback(partial(update5,new=message,old=source6_df,attr=source6.data))
            doc.add_next_tick_callback(partial(update4,new=message,old=source5_df,attr=data_table2.source))
        elif message.startswith('runtime'):
            print('---- Receive runtime statistics')
            doc.add_next_tick_callback(partial(update8,new=message,old=source8_df,attr=data_table4.source))
        elif message.startswith('global'):
            print('-----Receive global information')
            doc.add_next_tick_callback(partial(update7,new=message,old=source7_df,attr=data_table3.source))
        elif message.startswith('select'):
            print('---- Receive global price information')            
            doc.add_next_tick_callback(partial(update9,new=message,old=source9_df,attr=data_table5.source))
        elif message.startswith('price'):
            print('---- Receive local price information')            
            doc.add_next_tick_callback(partial(update10,new=message,old=source10_df,attr=data_table6.source))
        else: #start with each
            msg = message.split(' ')[1:]
            price_summary.loc[len(price_summary),:]=msg
            price_summary.to_csv('output_price.csv', index=False)


def retrieve_tasks(dag_info_file):
    config_file = open(dag_info_file,'r')
    dag_size = int(config_file.readline())

    G=nx.Graph()
    tasks={}
    for i, line in enumerate(config_file, 1):
        dag_line = line.strip().split(" ")
        tasks[dag_line[0]]=i 
        G.add_node(dag_line[0])
        nbs = dag_line[3:]
        for nb in nbs:
            if nb!='home':
                G.add_edge(dag_line[0],nb,weight=1) 
        if i == dag_size:
            break
    return tasks, G

def k8s_get_nodes(node_info_file):
    """read the node info from the file input
  
    Args:
        node_info_file (str): path of ``node.txt``
  
    Returns:
        dict: node information 
    """

    nodes = {}
    node_file = open(node_info_file, "r")
    compute_nodes = []
    home_nodes = []
    for i,line in enumerate(node_file):
        node_line = line.strip().split(" ")
        nodes[i] = [node_line[0],node_line[1]]    
        if node_line[0].startswith('home'):
            home_nodes.append(node_line[0])  
        else:
            compute_nodes.append(node_line[0])
    return nodes, home_nodes,compute_nodes


def repeatlist(it, count):
    return islice(cycle(it), count)

def update():
    m.client.loop(timeout=0.5)

@gen.coroutine
def update4(attr, old, new):
    assigned_info = new.split(' ')[1:]
    for info in assigned_info:
        tmp = info.split(':')
        new_source5_df.loc[new_source5_df.task_names==tmp[0],'assigned']=tmp[1]
        new_source5_df.loc[new_source5_df.task_names==tmp[0],'as_time']=convert_time(tmp[2])

    source5.data = {
        'task_id'       : new_source5_df.task_id,
        'task_names'    : new_source5_df.task_names,
        'assigned'      : new_source5_df.assigned,
        'as_time'       : new_source5_df.as_time
    }

@gen.coroutine
def update5(attr, old, new):
    assigned_info = new.split(' ')[1:]
    
    for info in assigned_info:
        tmp = info.split(':')
        t = '_T'+str(tasks[tmp[0]])
        n = 'N'+str(node_short.index(tmp[1]))
        new_source6_df.loc[new_source6_df.nodes==n,'assigned_task']=new_source6_df.loc[new_source6_df.nodes==n,'assigned_task']+t
    source6.data = {
        'x' : new_source6_df.x,
        'y' : new_source6_df.y,
        'color':new_source6_df.color,
        'nodes':new_source6_df.nodes,
        'x_label':new_source6_df.x_label,
        'y_label':new_source6_df.y_label,
        'assigned_task':new_source6_df.assigned_task    
    }

@gen.coroutine
def update7(attr, old, new):

    tmp = new.split(' ')[1:]
    home_id = tmp[0]
    if tmp[1]=='start':
        data = ['N/A','N/A',tmp[2],tmp[0],convert_time(tmp[3])]
        new_source7_df.loc[len(new_source7_df),:]=data
    else:
        new_source7_df.loc[(new_source7_df.home_id==tmp[0]) & (new_source7_df.global_input==tmp[2]),'end_times']=convert_time(tmp[3])
        tmp1 = new_source7_df.loc[(new_source7_df.home_id==tmp[0]) & (new_source7_df.global_input==tmp[2]),'end_times']
        tmp2 = new_source7_df.loc[(new_source7_df.home_id==tmp[0]) & (new_source7_df.global_input==tmp[2]),'start_times'] 
        new_source7_df.loc[(new_source7_df.home_id==tmp[0]) & (new_source7_df.global_input==tmp[2]),'exec_times']=time_delta(tmp1,tmp2)
    source7.data = {
        'home_id'       : new_source7_df.home_id,
        'global_input'  : new_source7_df.global_input,
        'start_times'   : new_source7_df.start_times,
        'end_times'     : new_source7_df.end_times,
        'exec_times'    : new_source7_df.exec_times,
    }
    source7.data.to_csv('output_global.csv')


@gen.coroutine
def update8(attr, old, new):
    tmp = new.split(' ')[1:]

    if tmp[0]=='enter':
        data = [tmp[1],'N/A','N/A',convert_time(tmp[5]),'N/A','N/A',tmp[4],'N/A',tmp[2],tmp[3]]
        new_source8_df.loc[len(new_source8_df),:]=data    
    elif tmp[0]=='exec':
        new_source8_df.loc[(new_source8_df.home_ids==tmp[1]) & (new_source8_df.task_name==tmp[3]) & (new_source8_df.local_input==tmp[4]),'local_exec_times']=convert_time(tmp[5])
        tmp1 = new_source8_df.loc[(new_source8_df.home_ids==tmp[1]) & (new_source8_df.task_name==tmp[3]) & (new_source8_df.local_input==tmp[4]),'local_exec_times']
        tmp2 = new_source8_df.loc[(new_source8_df.home_ids==tmp[1]) & (new_source8_df.task_name==tmp[3]) & (new_source8_df.local_input==tmp[4]),'local_enter_times']
        new_source8_df.loc[(new_source8_df.home_ids==tmp[1]) & (new_source8_df.task_name==tmp[3]) & (new_source8_df.local_input==tmp[4]),'local_waiting_times']=time_delta(tmp1,tmp2) 
        new_source8_df.loc[(new_source8_df.home_ids==tmp[1]) & (new_source8_df.task_name==tmp[3]) & (new_source8_df.local_input==tmp[4]),'node_name']=tmp[2]   
    else:
        new_source8_df.loc[(new_source8_df.home_ids==tmp[1]) & (new_source8_df.task_name==tmp[3]) & (new_source8_df.local_input==tmp[4]),'local_finish_times']=convert_time(tmp[5])
        tmp1 = new_source8_df.loc[(new_source8_df.home_ids==tmp[1]) & (new_source8_df.task_name==tmp[3]) & (new_source8_df.local_input==tmp[4]),'local_finish_times']
        tmp2 = new_source8_df.loc[(new_source8_df.home_ids==tmp[1]) & (new_source8_df.task_name==tmp[3]) & (new_source8_df.local_input==tmp[4]),'local_enter_times']
        tmp3 = new_source8_df.loc[(new_source8_df.home_ids==tmp[1]) & (new_source8_df.task_name==tmp[3]) & (new_source8_df.local_input==tmp[4]),'local_exec_times']
        new_source8_df.loc[(new_source8_df.home_ids==tmp[1]) & (new_source8_df.task_name==tmp[3]) & (new_source8_df.local_input==tmp[4]),'local_elapse_times']=time_delta(tmp1,tmp2)   
        new_source8_df.loc[(new_source8_df.home_ids==tmp[1]) & (new_source8_df.task_name==tmp[3]) & (new_source8_df.local_input==tmp[4]),'local_duration_times']=time_delta(tmp1,tmp3) 

    source8.data = {
        'home_ids'              : new_source8_df.home_ids,
        'node_name'             : new_source8_df.node_name,
        'local_duration_times'  : new_source8_df.local_duration_times,
        'local_elapse_times'    : new_source8_df.local_elapse_times,
        'local_enter_times'     : new_source8_df.local_enter_times,
        'local_exec_times'      : new_source8_df.local_exec_times,
        'local_finish_times'    : new_source8_df.local_finish_times,
        'local_input'           : new_source8_df.local_input,
        'local_waiting_times'   : new_source8_df.local_waiting_times,
        'task_name'             : new_source8_df.task_name
    }

@gen.coroutine
def update9(attr, old, new):
    tmp = new.split(' ')[1:]
    new_source9_df.loc[(new_source9_df['home_ids'] == tmp[0]) & (new_source9_df['task_name'] == tmp[1]),'best_compute'] = tmp[2]
    new_source9_df.loc[(new_source9_df['home_ids'] == tmp[0]) & (new_source9_df['task_name'] == tmp[1]),'best_price'] = tmp[3]
    new_source9_df.loc[(new_source9_df['home_ids'] == tmp[0]) & (new_source9_df['task_name'] == tmp[1]),'updated_time'] = convert_time(tmp[4])
    source9.data = {
        'home_ids'              : new_source9_df.home_ids,
        'task_name'             : new_source9_df.task_name,
        'best_compute'          : new_source9_df.best_compute,
        'best_price'            : new_source9_df.best_price,
        'updated_time'          : new_source9_df.updated_time
    }
    data = [tmp[2],tmp[3],tmp[0],tmp[1],convert_time(tmp[4]),tmp[5]]
    assign_summary.loc[len(assign_summary),:]=data    
    # print(assign_summary)
    assign_summary.to_csv('output_assign.csv', index=False)



@gen.coroutine
def update10(attr, old, new):
    tmp = new.split(' ')[1:] 
    print(tmp)
    task_name = tmp[0]
    pricing_info = tmp[1].split('#')
    node_name = pricing_info[0]
    price_cpu = pricing_info[1]
    price_mem = pricing_info[2]
    price_queue = pricing_info[3].split('$')[0]
    price_net_info = pricing_info[3].split('$')[1:]
    price_net = ""
    for price in price_net_info:
        price_net = price_net+price.split('%')[0]+' '
        price_net = price_net+price.split('%')[1]+' '
    
    updated_time = convert_time(tmp[2])
    new_source10_df.loc[(new_source10_df['node_name'] == node_name) & (new_source10_df['task_name'] == task_name),'price_cpu'] = price_cpu
    new_source10_df.loc[(new_source10_df['node_name'] == node_name) & (new_source10_df['task_name'] == task_name),'price_mem'] = price_mem
    new_source10_df.loc[(new_source10_df['node_name'] == node_name) & (new_source10_df['task_name'] == task_name),'price_queue'] = price_queue
    new_source10_df.loc[(new_source10_df['node_name'] == node_name) & (new_source10_df['task_name'] == task_name),'price_net'] = price_net
    new_source10_df.loc[(new_source10_df['node_name'] == node_name) & (new_source10_df['task_name'] == task_name),'updated_time'] = updated_time    

    source10.data = {
        'node_name'             : new_source10_df.node_name,
        'price_cpu'             : new_source10_df.price_cpu,
        'price_mem'             : new_source10_df.price_mem,
        'price_net'             : new_source10_df.price_net,
        'price_queue'           : new_source10_df.price_queue,
        'task_name'             : new_source10_df.task_name,
        'updated_time'          : new_source10_df.updated_time
    }

    

def convert_time(t):
    return datetime.datetime.fromtimestamp(float(t)).strftime("%d.%m.%y %H:%M:%S")
def time_delta(end,start): 
    tmp1 = datetime.datetime.strptime(end.iloc[0],"%d.%m.%y %H:%M:%S")
    tmp2 = datetime.datetime.strptime(start.iloc[0],"%d.%m.%y %H:%M:%S")
    delta = (tmp1-tmp2).total_seconds()
    return delta

###################################################################################################



global OUTFNAME, SERVER_IP, SUBSCRIPTIONS, DAG_PATH,NODE_PATH
OUTFNAME = 'demo.html'
SERVER_IP = "127.0.0.1"
SUBSCRIPTIONS = 'JUPITER'
APP_PATH = '../app_specific_files/dummy_app/'
#APP_PATH = '../app_specific_files/network_monitoring_app_dag/'
DAG_PATH = APP_PATH+'configuration.txt'
NODE_PATH = '../nodes.txt'

global start_time, finish_time, total_time, offset, input_num
start_time =[]
finish_time =0
total_time =0
offset = 0
input_num = 0

global source, source3,source4,source5,source6, source5_df, doc, nodes, m, p,p1

source = ColumnDataSource(data=dict(top=[0], bottom=[0],left=[0],right=[0], color=["#9ecae1"],line_color=["black"], line_width=[2]))


global nodes, home_nodes,compute_nodes, num_nodes,MAX_X,MAX_Y,tasks, G
nodes, home_nodes,compute_nodes = k8s_get_nodes(NODE_PATH)
num_nodes = len(nodes)
MAX_X = 10
MAX_Y = 12
tasks, G = retrieve_tasks(DAG_PATH)
num_tasks = len(tasks)

doc = curdoc()
doc.title = 'CIRCE Visualization'

m = mq(outfname=OUTFNAME,subs=SUBSCRIPTIONS,server = SERVER_IP,port=1883,timeout=60,looptimeout=1)




###################################################################################################################################

global data_table, data_table2, source5_df,new_source5_df,source6_df,new_source6_df

node_id = ['N'+str(i) for i in nodes.keys()]
node_short = [i[0] for i in nodes.values()]
node_full = [i[1] for i in nodes.values()]

source4 = ColumnDataSource(dict(node_id=node_id,node_short=node_short,node_full=node_full))
columns = [TableColumn(field="node_id", title="Node ID"),TableColumn(field="node_short", title="Node Name"),TableColumn(field="node_full", title="Full Name")]
data_table = DataTable(source=source4, columns=columns, width=400, height=230,
                       selectable=True)

title1 = Div(text='Node Information',style={'font-size': '15pt', 'color': 'black','text-align': 'center'},width=400, height=20)

assignment = ['N/A']*len(tasks.keys())
assign_time = ['N/A']*len(tasks.keys())
task_id = ['T'+str(i) for i in tasks.values()]
source5 = ColumnDataSource(data=dict(task_id=task_id,task_names=list(tasks.keys()),assigned = assignment,as_time=assign_time))
columns2 = [TableColumn(field="task_id", title="Task ID"),TableColumn(field="task_names", title="Tasks Names"),TableColumn(field="assigned", title="Assigned Node"),TableColumn(field="as_time", title="Assigned Time")]
data_table2 = DataTable(source=source5, columns=columns2, width=400, height=230,
                       selectable=True,editable=True)
title2 = Div(text='Task Controller Mapping Information',style={'font-size': '15pt', 'color': 'black','text-align': 'center'},width=400, height=20)
source5_df=source5.to_df()
new_source5_df=source5.to_df()
data_table2.on_change('source', lambda attr, old, new: update4())

###################################################################################################################################

points = set()
while len(points) < num_nodes:
    ix = np.random.randint(1, MAX_X)
    iy = np.random.randint(1, MAX_Y)
    points.add((ix,iy))
x = [i[0] for i in points]
y = [i[1] for i in points]
x_label = [i - 0.3 for i in x]
y_label = [i - 0.2 for i in y]
c = brewer["Spectral"][9]
color = list(repeatlist(c, num_nodes))
assigned_task=[""]*len(points)
p = figure(x_range=(0, MAX_X+1), y_range=(0, MAX_Y+1),plot_width=500, plot_height=600)
p.background_fill_color = "#EEEDED"
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
p.xaxis.axis_label = 'Digital Ocean Clusters'
p.xaxis.axis_label_text_font_size='20pt'


source6 = ColumnDataSource(data=dict(x=x,y=y,color=color,nodes=node_id,x_label=x_label,y_label=y_label,assigned_task=assigned_task))

lab = LabelSet(x='x_label', y='y_label',text='nodes',text_font_style='bold',text_color='black',source=source6)
p.add_layout(lab)
w = p.circle( x='x',y='y', radius=0.8, fill_color='color',line_color='color',source=source6)

p.add_tools(HoverTool(
    tooltips=[
        ("node_id","@nodes"),
        ("assigned_task","@assigned_task"),
    ]
))

source6_df = source6.to_df()
new_source6_df = source6.to_df()


###################################################################################################################################

node_name = list(G.nodes())
num_nodes = len(node_name)
positions = nx.spring_layout(G)
c = brewer["Spectral"][9]
colors = list(repeatlist(c, num_nodes))
nx.set_node_attributes(G, colors, 'colors')

source11=ColumnDataSource(pd.DataFrame.from_dict({k:v for k,v in G.nodes(data=True)},orient='index'))
start_edge = [start_edge for (start_edge, end_edge) in G.edges()]
end_edge = [end_edge for (start_edge, end_edge) in G.edges()]
weight = list(nx.get_edge_attributes(G,'weight').values())
edge_df = pd.DataFrame({'source':start_edge, 'target':end_edge, 'weight':weight})
G = nx.from_pandas_edgelist(edge_df,edge_attr=True)
graph = from_networkx(G, nx.spring_layout, scale=2, center=(0,0))
p1 = figure(x_range=(-1.1,1.1), y_range=(-1.1,1.1),tools = "pan,wheel_zoom,box_select,reset,box_zoom,crosshair",plot_width=600,plot_height=600)
graph.node_renderer.data_source = source11
graph.node_renderer.data_source.data['colors'] = colors
graph.node_renderer.glyph = Circle(size=30,fill_color='colors')
graph.layout_provider = StaticLayoutProvider(graph_layout=positions)
graph.selection_policy = NodesOnly()
p1.renderers.append(graph)
p1.tools.append(HoverTool(
    tooltips=[
    ('Name', '@index'),
    ('Weight','@weight')
    ]
    ))

###################################################################################################################################

global data_table3, data_table4, source7_df,new_source7_df,source8_df,new_source8_df

home_id = []
global_input = []
start_times = []
end_times = []
exec_times = []
source7 = ColumnDataSource(dict(home_id=home_id,global_input=global_input,start_times=start_times,end_times=end_times,exec_times=exec_times))
columns = [TableColumn(field="home_id", title="Home ID"),TableColumn(field="global_input", title="Global Input Name"),TableColumn(field="start_times", title="Enter time"),TableColumn(field="end_times", title="Finish tme"),TableColumn(field="exec_times", title="Make span [s]")]
data_table3 = DataTable(source=source7, columns=columns, width=600, height=580,selectable=True)

title3 = Div(text='Global Input Information',style={'font-size': '15pt', 'color': 'black','text-align': 'center'},width=600, height=20)

source7_df=source7.to_df()
new_source7_df=source7.to_df()
data_table3.on_change('source', lambda attr, old, new: update7())

home_ids = []
node_name = []
task_name = []
local_input = []
local_enter_times = []
local_exec_times = []
local_finish_times = []
local_elapse_times = []
local_duration_times = []
local_waiting_times = []

source8 = ColumnDataSource(dict(home_ids=home_ids,node_name=node_name,task_name=task_name,local_input=local_input,local_enter_times=local_enter_times,local_exec_times=local_exec_times,local_finish_times=local_finish_times,local_elapse_times=local_elapse_times,local_duration_times=local_duration_times,local_waiting_times=local_waiting_times))
columns = [TableColumn(field="home_ids", title="Home ID"),TableColumn(field="node_name", title="Compute node"),TableColumn(field="task_name", title="Task name"),TableColumn(field="local_input", title="Local Input"),
            TableColumn(field="local_enter_times", title="Enter time"),TableColumn(field="local_exec_times", title="Exec time"),
            TableColumn(field="local_finish_times", title="Finish time"),TableColumn(field="local_elapse_times", title="Elapse time"),
            TableColumn(field="local_duration_times", title="Duration time"),TableColumn(field="local_waiting_times", title="Waiting time")]
data_table4 = DataTable(source=source8, columns=columns, width=900, height=580,selectable=True)

title4 = Div(text='Local Input Information',style={'font-size': '15pt', 'color': 'black','text-align': 'center'},width=600, height=20)

source8_df=source8.to_df()
new_source8_df=source8.to_df()
data_table4.on_change('source', lambda attr, old, new: update8())

###################################################################################################################################
global data_table5, source9_df,new_source9_df,assign_summary

task_list = list(tasks.keys())
tmp = list(itertools.product(home_nodes, task_list))

home_ids = [x[0] for x in tmp]
task_name = [x[1] for x in tmp]
best_compute = ['N/A'] * len(home_ids)
best_price = ['N/A'] * len(home_ids)
updated_time = ['N/A'] * len(home_ids)

source9 = ColumnDataSource(dict(home_ids=home_ids,task_name=task_name,best_compute=best_compute,best_price=best_price,updated_time=updated_time))
columns = [TableColumn(field="home_ids", title="Home ID"),TableColumn(field="task_name", title="Task name"),TableColumn(field="best_compute", title="Best compute node"),
            TableColumn(field="best_price", title="Best price"),TableColumn(field="updated_time", title="Updated time")]
data_table5 = DataTable(source=source9, columns=columns, width=600, height=300,selectable=True)
title5 = Div(text='Global Price Information',style={'font-size': '15pt', 'color': 'black','text-align': 'center'},width=600, height=20)

source9_df=source9.to_df()
new_source9_df=source9.to_df()
assign_summary = pd.DataFrame(columns=['best_compute', 'best_price', 'home_ids', 'task_name', 'updated_time','file_name'])

data_table5.on_change('source', lambda attr, old, new: update9())

global data_table6, source10_df,new_source10_df, price_summary

tmp = list(itertools.product(home_nodes, task_list,compute_nodes))

home_ids = [x[0] for x in tmp]
task_name = [x[1] for x in tmp]
node_name = [x[2] for x in tmp]
price_cpu = ['N/A'] * len(home_ids)
price_mem = ['N/A'] * len(home_ids)
price_net = ['N/A'] * len(home_ids)
price_queue = ['N/A'] * len(home_ids)
updated_time = ['N/A'] * len(home_ids)

source10 = ColumnDataSource(dict(task_name=task_name,node_name=node_name,price_cpu=price_cpu,price_mem=price_mem,price_net=price_net,price_queue=price_queue,updated_time=updated_time))
columns = [TableColumn(field="task_name", title="Task name"),TableColumn(field="node_name", title="Node name"),
           TableColumn(field="price_cpu", title="CPU Price"),TableColumn(field="price_mem", title="Memory Price"),TableColumn(field="price_net", title="Network Price"),
           TableColumn(field="price_queue", title="Queue Price"),TableColumn(field="updated_time", title="Updated time")]
data_table6 = DataTable(source=source10, columns=columns, width=900, height=300,selectable=True)
title6 = Div(text='Local Price Information',style={'font-size': '15pt', 'color': 'black','text-align': 'center'},width=900, height=20)

source10_df=source10.to_df()
new_source10_df=source10.to_df()
price_summary = pd.DataFrame(columns=['home_id','task_name','compute_node','file_name','price_cpu','price_mem','price_queue','price_net','update_time'])
               
data_table6.on_change('source', lambda attr, old, new: update10())


###################################################################################################################################
p2 = layout([title1,widgetbox(data_table,width=400,height=280),title2,widgetbox(data_table2,width=400,height=280)],sizing_mode='fixed',width=400,height=600)
layout = row(p2,p,p1)
# layout = row(p2,p)
doc.add_root(layout)

p5 = column(title5,widgetbox(data_table5))
p6 = column(title6,widgetbox(data_table6))
doc.add_root(row(p5,p6))
p3 = column(title3,widgetbox(data_table3))
p4 = column(title4,widgetbox(data_table4))
doc.add_root(row(p3,p4))
doc.add_periodic_callback(update, 50) 