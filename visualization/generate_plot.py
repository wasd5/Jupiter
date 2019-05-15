import pandas as pd
import datetime
import matplotlib.pyplot as plt





############################################################################################################################################

# N = range(50)
# files = ['%d'%(i+1) for i in N]
# latency = [33,32,33,32,32,32,32,32,32,32,32,32,32,32,36,33,33,32,32,32,36,32,32,32,32,32,33,32,41,40,41,40,37,39,39,40,40,40,40,39,39,39,39,41,39,40,39,39,40,40]
# plt.plot(files,latency)
# plt.axvline(x='26',color = 'k',linestyle='--')
# plt.title('Make span for dummy application (pricing-event scheme)',fontsize=24)
# plt.xlabel('Incoming files', fontsize=14)
# plt.ylabel('Latency (sec)', fontsize=14)
# plt.show()

############################################################################################################################################


# pricing_file = 'output_price.csv'
# pricing_summary = pd.read_csv(pricing_file)
# pricing_summary['ID'] = pricing_summary.apply(lambda row: row.file_name.split('botnet')[0], axis=1)
# w_net = 1 # Network profiling: longer time, higher price
# w_cpu = 10000 # Resource profiling : larger cpu resource, lower price
# w_mem = 10000 # Resource profiling : larger mem resource, lower price
# w_queue = 1 # Queue : currently 0
# pricing_summary['total_price'] = pricing_summary.apply(lambda row: row.price_cpu*w_cpu+row.price_mem*w_mem+row.price_net*w_net+row.price_queue*w_queue, axis=1)
# task_list = pricing_summary.task_name.unique()
# nodes = pricing_summary.compute_node.unique()
# ID = pricing_summary.ID.unique()
# colors = ['b','g','r']
 


# for task in task_list:
# 	subprice = pricing_summary.loc[pricing_summary.task_name==task,:]
# 	fig, ax = plt.subplots()
# 	for idx,node in enumerate(nodes):
# 		p = subprice.loc[subprice.compute_node==node,'total_price']
# 		ID = subprice.loc[subprice.compute_node==node,'ID']
# 		#plt.plot(ID,p,color=colors[idx])
# 		ax.plot(ID, p, color=colors[idx], label=node)
# 	leg = ax.legend()
# 	plt.axvline(x='26',color = 'k',linestyle='--')
# 	title = 'Price for dummy application - %s'%task
# 	plt.title(title,fontsize=24)
# 	plt.xlabel('Incoming files', fontsize=14)
# 	ylabel = 'Price: w_net=%d,w_cpu=%d,w_mem=%d'%(w_net,w_cpu,w_mem)
# 	plt.ylabel(ylabel, fontsize=14)
# 	plt.show()
	
# 	# name = 'dummy_'+task+'.png'
# 	# plt.savefig(name)


# Index(['home_id', 'task_name', 'compute_node', 'file_name', 'price_cpu',
#        'price_mem', 'price_queue', 'price_net', 'update_time'],

############################################################################################################################################

# assignment_file = 'output_assign.csv'
# assign_summary = pd.read_csv(assignment_file)
# print(assign_summary)
# assign_summary['ID'] = assign_summary.apply(lambda row: row.file_name.split('botnet')[0], axis=1)
# plt.plot(assign_summary['ID'],assign_summary['best_compute'])
# plt.axvline(x='26',color = 'k',linestyle='--')
# plt.title('Best compute node selection for dummy application (pricing-event scheme)',fontsize=24)
# plt.xlabel('Incoming files', fontsize=14)
# plt.ylabel('Best compute node', fontsize=14)
# plt.show()

# Index(['best_compute', 'best_price', 'home_ids', 'task_name', 'updated_time',
#        'file_name'],


############################################################################################################################################






# fig, ax1 = plt.subplots()


# ax2 = ax1.twinx()

# pricing_file = 'output_price.csv'
# pricing_summary = pd.read_csv(pricing_file)
# pricing_summary['ID'] = pricing_summary.apply(lambda row: row.file_name.split('botnet')[0], axis=1)
# w_net = 1 # Network profiling: longer time, higher price
# w_cpu = 10000 # Resource profiling : larger cpu resource, lower price
# w_mem = 10000 # Resource profiling : larger mem resource, lower price
# w_queue = 1 # Queue : currently 0
# pricing_summary['total_price'] = pricing_summary.apply(lambda row: row.price_cpu*w_cpu+row.price_mem*w_mem+row.price_net*w_net+row.price_queue*w_queue, axis=1)
# task_list = pricing_summary.task_name.unique()
# nodes = pricing_summary.compute_node.unique()
# ID = pricing_summary.ID.unique()
# colors = ['b','g','r']

# task = 'task2'
# subprice = pricing_summary.loc[pricing_summary.task_name==task,:]
# for idx,node in enumerate(nodes):
# 	p = subprice.loc[subprice.compute_node==node,'total_price']
# 	ID = subprice.loc[subprice.compute_node==node,'ID']
# 	ax1.plot(ID, p, color=colors[idx], label=node)
# leg = ax1.legend()
# ax1.axvline(x='26',color = 'k',linestyle='--')
# ylabel = 'Price: w_net=%d,w_cpu=%d,w_mem=%d'%(w_net,w_cpu,w_mem)
# ax1.set_ylabel(ylabel, fontsize=14,color='k')

#################################

# fig, ax2 = plt.subplots()


# ax1 = ax2.twinx()

# assignment_file = 'output_assign.csv'
# assign_summary = pd.read_csv(assignment_file)
# assign_summary['ID'] = assign_summary.apply(lambda row: row.file_name.split('botnet')[0], axis=1)
# ax2.plot(assign_summary['ID'],assign_summary['best_compute'],'co')
# ax2.set_title('Makespan for dummy application',fontsize=24)
# ax2.set_xlabel('Incoming files', fontsize=14)
# ax2.set_ylabel('Best compute node (pricing-event)', fontsize=14,color='c')

# ax2.axvline(x='26',color = 'k',linestyle='--')



# N = range(50)
# files = ['%d'%(i+1) for i in N]
# latency = [33,32,33,32,32,32,32,32,32,32,32,32,32,32,36,33,33,32,32,32,36,32,32,32,32,32,33,32,41,40,41,40,37,39,39,40,40,40,40,39,39,39,39,41,39,40,39,39,40,40]
# latency_original = [23,23,23,23,23,23,23, 22,22,22,22,24,22,23,22,22,22,22,22,23,23,22,22,23,23,23,23,23,22,23,23,23,23,23,23,24,23,24,24,24,22,23,22,23,23,23,24,25,22,23]
# ax1.plot(files,latency, 'b',label='pricing')
# ax1.plot(files,latency_original, 'g',label='non-pricing')
# #ax1.set_xlabel('Incoming files', fontsize=14)
# ax1.set_ylabel('Latency (sec)', fontsize=14,color='r')
# leg = ax1.legend(loc='upper left')

# plt.show()


N = range(50)
files = ['%d'%(i+1) for i in N]
latency = [33,32,33,32,32,32,32,32,32,32,32,32,32,32,36,33,33,32,32,32,36,32,32,32,32,32,33,32,41,40,41,40,37,39,39,40,40,40,40,39,39,39,39,41,39,40,39,39,40,40]
latency_original = [23,23,23,23,23,23,23, 22,22,22,22,24,22,23,22,22,22,22,22,23,23,22,22,23,23,23,23,23,22,23,23,23,23,23,23,24,23,24,24,24,22,23,22,23,23,23,24,25,22,23]
plt.plot(files,latency,'b',label='pricing')
plt.plot(files,latency_original, 'g',label='non-pricing')
plt.axvline(x='26',color = 'k',linestyle='--')
plt.title('Make span for dummy application (pricing-event scheme versus non-pricing)',fontsize=24)
plt.xlabel('Incoming files', fontsize=14)
plt.ylabel('Latency (sec)', fontsize=14)
# plt.show()

plt.axvline(x='26',color = 'k',linestyle='--')



# N = range(50)
# files = ['%d'%(i+1) for i in N]
# latency = [33,32,33,32,32,32,32,32,32,32,32,32,32,32,36,33,33,32,32,32,36,32,32,32,32,32,33,32,41,40,41,40,37,39,39,40,40,40,40,39,39,39,39,41,39,40,39,39,40,40]
# latency_original = [23,23,23,23,23,23,23, 22,22,22,22,24,22,23,22,22,22,22,22,23,23,22,22,23,23,23,23,23,22,23,23,23,23,23,23,24,23,24,24,24,22,23,22,23,23,23,24,25,22,23]
# ax1.plot(files,latency, 'b',label='pricing')
# ax1.plot(files,latency_original, 'g',label='non-pricing')
# #ax1.set_xlabel('Incoming files', fontsize=14)
# ax1.set_ylabel('Latency (sec)', fontsize=14,color='r')
leg = plt.legend(loc='upper left')

plt.show()