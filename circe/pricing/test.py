graph = dict()
graph = {'task0': {'task2', 'task3', 'task1', 'task4'}, 'task3': {'task6', 'task4', 'task5'}, 'task6': {'home'}, 'task2': {'task6', 'task4', 'task5'}, 'task1': {'task4'}, 'task4': {'task6'}, 'task5': {'task6'}}
# task 6 -> home ????

print(graph)
glocal_task_node_map = dict()

from collections import defaultdict 
  
#Class to represent a graph 

local_task_node_map = {('node14', 'task6'): 'node10', ('node3', 'task6'): 'node10', ('node14', 'task0'): 'node10', ('node13', 'task1'): 'node10', ('node9', 'task1'): 'node10', ('node13', 'task5'): 'node10', ('node15', 'task4'): 'node10', ('node14', 'task4'): 'node10', ('node9', 'task6'): 'node10', ('node10', 'task0'): 'node6', ('node10', 'task3'): 'node6', ('node6', 'task5'): 'node10', ('node14', 'task5'): 'node10', ('node11', 'task1'): 'node10', ('node15', 'task1'): 'node10', ('node18', 'task4'): 'node10', ('node10', 'task2'): 'node6', ('node6', 'task3'): 'node10', ('node11', 'task2'): 'node10', ('node8', 'task5'): 'node10', ('node11', 'task3'): 'node10', ('node3', 'task0'): 'node10', ('node3', 'task1'): 'node10', ('node8', 'task4'): 'node10', ('node6', 'task6'): 'node10', ('node8', 'task6'): 'node10', ('home', 'task5'): 'node10', ('node15', ''): 'node10', ('node3', 'task4'): 'node10', ('node9', 'task0'): 'node10', ('node8', 'task1'): 'node10', ('node15', 'task5'): 'node10', ('node8', 'task3'): 'node10', ('node11', 'task5'): 'node10', ('node8', 'task2'): 'node10', ('node18', 'task3'): 'node10', ('node3', 'task5'): 'node10', ('node15', 'task6'): 'node10', ('node11', 'task4'): 'node10', ('home', ''): 'node10', ('node12', 'task6'): 'node10', ('home', 'task3'): 'node10', ('home', 'task1'): 'node10', ('node18', 'task5'): 'node10', ('node12', ''): 'node10', ('node13', ''): 'node10', ('node6', 'task0'): 'node10', ('node6', 'task2'): 'node10', ('node14', 'task2'): 'node10', ('node10', 'task4'): 'node6', ('node9', 'task3'): 'node10', ('node10', 'task5'): 'node6', ('node6', 'task4'): 'node10', ('node13', 'task0'): 'node10', ('node10', 'task1'): 'node6', ('node13', 'task2'): 'node10', ('node12', 'task4'): 'node10', ('node18', 'task6'): 'node10', ('node14', 'task3'): 'node10', ('node9', 'task5'): 'node10', ('node13', 'task4'): 'node10', ('home', 'task4'): 'node10', ('node11', ''): 'node10', ('node11', 'task0'): 'node10', ('node6', ''): 'node10', ('node9', ''): 'node10', ('node12', 'task5'): 'node10', ('node3', 'task2'): 'node10', ('home', 'task0'): 'node10', ('node3', 'task3'): 'node10', ('node3', ''): 'node10', ('node8', 'task0'): 'node10', ('node12', 'task0'): 'node10', ('node14', 'task1'): 'node10', ('node9', 'task2'): 'node10', ('node9', 'task4'): 'node10', ('node13', 'task6'): 'node10', ('node18', 'task2'): 'node10', ('node8', ''): 'node10', ('node18', 'task0'): 'node10', ('node18', ''): 'node10', ('node12', 'task2'): 'node10', ('node10', 'task6'): 'node6', ('home', 'task2'): 'node10', ('node18', 'task1'): 'node10', ('node11', 'task6'): 'node10', ('node15', 'task0'): 'node10', ('node13', 'task3'): 'node10', ('node14', ''): 'node10', ('node15', 'task2'): 'node10', ('home', 'task6'): 'node10', ('node6', 'task1'): 'node10', ('node12', 'task1'): 'node10', ('node15', 'task3'): 'node10', ('node12', 'task3'): 'node10'}
class Graph(): 
    def __init__(self,vertices): 
        self.graph = defaultdict(list) #dictionary containing adjacency List 
        self.V = vertices #List of vertices 
  
    # function to add an edge to graph 
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
    def topologicalSort(self): 
              
        #in_degree = [0]*(self.V) 
        in_degree = dict.fromkeys(self.V, 0)
          
        # Traverse adjacency lists to fill indegrees of 
           # vertices.  This step takes O(V+E) time 
        for i in self.graph: 
            for j in self.graph[i]:
                in_degree[j] += 1

        # Create an queue and enqueue all vertices with 
        # indegree 0 
        queue = [] 
        # for i in range(self.V):
        for i in self.V:
            if in_degree[i] == 0:
                queue.append(i) 

        #Initialize count of visited vertices 
        cnt = 0

        # Create a vector to store result (A topological 
        # ordering of the vertices) 
        top_order = [] 

        # One by one dequeue vertices from queue and enqueue 
        # adjacents if indegree of adjacent becomes 0 
        while queue: 

            # Extract front of queue (or perform dequeue) 
            # and add it to topological order 
            u = queue.pop(0) 
            top_order.append(u) 
            # Iterate through all neighbouring nodes 
            # of dequeued node u and decrease their in-degree 
            # by 1 
            for i in self.graph[u]: 
                in_degree[i] -= 1
                # If in-degree becomes zero, add it to queue 
                if in_degree[i] == 0: 
                    queue.append(i)
            cnt += 1

        # Check if there was a cycle 
        if cnt != len(self.V): 
            print("There exists a cycle in the graph")
        else : 
            #Print topological order 
            print(top_order) 

g= Graph(['0','1','2','3','4','5','6']) 
g.addEdge('0', '1'); 
g.addEdge('0', '2'); 
g.addEdge('0', '3'); 
g.addEdge('1', '4'); 
g.addEdge('3', '4'); 
g.addEdge('2', '4');
g.addEdge('2','5'); 
g.addEdge('3', '5');
g.addEdge('3', '6');
g.addEdge('4', '6'); 
g.addEdge('5', '6');

g.topologicalSort() 

#{'task0': ['task1', 'task3', 'task2', 'task4'], 'task3': ['task5', 'task4', 'task6'], 'task6': [], 'task2': ['task5', 'task4', 'task6'], 'task1': ['task4'], 'task4': ['task6'], 'task5': ['task6']}

