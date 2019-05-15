# import pandas as pd
# import numpy as np

# from bokeh.layouts import row, widgetbox, column
# from bokeh.models import ColumnDataSource, CustomJS, StaticLayoutProvider, Oval, Circle
# from bokeh.models import HoverTool, TapTool, BoxSelectTool, GraphRenderer
# from bokeh.models.widgets import RangeSlider, Button, DataTable, TableColumn, NumberFormatter
# from bokeh.io import curdoc, show, output_notebook
# from bokeh.plotting import figure

# import networkx as nx

# from bokeh.io import show, output_file
# from bokeh.plotting import figure
# from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes, NodesOnly

# # Import / instantiate networkx graph
# G = nx.Graph()

# G.add_edge('a', 'b', weight=0.6)
# G.add_edge('a', 'c', weight=0.2)
# G.add_edge('c', 'd', weight=0.1)
# G.add_edge('c', 'e', weight=0.7)
# G.add_edge('c', 'f', weight=0.9)
# G.add_edge('a', 'd', weight=0.3)

# # Node Characteristics
# node_name = list(G.nodes())
# positions = nx.spring_layout(G)

# node_size = [k*4 for k in range(len(G.nodes()))]
# nx.set_node_attributes(G, node_size, 'node_size')
# visual_attributes=ColumnDataSource(
#     pd.DataFrame.from_dict({k:v for k,v in G.nodes(data=True)},orient='index'))

# # Edge characteristics
# start_edge = [start_edge for (start_edge, end_edge) in G.edges()]
# end_edge = [end_edge for (start_edge, end_edge) in G.edges()]
# weight = list(nx.get_edge_attributes(G,'weight').values())

# edge_df = pd.DataFrame({'source':start_edge, 'target':end_edge, 'weight':weight})

# # Create full graph from edgelist 
# G = nx.from_pandas_edgelist(edge_df,edge_attr=True)

# # Convert full graph to Bokeh network for node coordinates and instantiate Bokeh graph object 
# G_source = from_networkx(G, nx.spring_layout, scale=2, center=(0,0))
# graph = GraphRenderer()

# # # Update loop where the magic happens
# # def update():
# #     selected_df = edge_df[(edge_df['weight'] >= slider.value[0]) & (edge_df['weight'] <= slider.value[1])]
# #     sub_G = nx.from_pandas_edgelist(selected_df,edge_attr=True)
# #     sub_graph = from_networkx(sub_G, nx.spring_layout, scale=2, center=(0,0))
# #     graph.edge_renderer.data_source.data = sub_graph.edge_renderer.data_source.data
# #     graph.node_renderer.data_source.data = G_source.node_renderer.data_source.data
# #     graph.node_renderer.data_source.add(node_size,'node_size')

# # def selected_points(attr,old,new):
# #     selected_idx = graph.node_renderer.selected.indices #does not work
# #     print(selected_idx)

# # Slider which changes values to update the graph
# # slider = RangeSlider(title="Weights", start=0, end=1, value=(0.25, 0.75), step=0.10)
# # slider.on_change('value', lambda attr, old, new: update())

# # Plot object which is updated 
# plot = figure(title="Meetup Network Analysis", x_range=(-1.1,1.1), y_range=(-1.1,1.1),
#              tools = "pan,wheel_zoom,box_select,reset,box_zoom,crosshair", plot_width=800, plot_height=800)

# # Assign layout for nodes, render graph, and add hover tool
# graph.layout_provider = StaticLayoutProvider(graph_layout=positions)
# graph.node_renderer.glyph = Circle(size='node_size')
# graph.selection_policy = NodesOnly()
# plot.renderers.append(graph)
# plot.tools.append(HoverTool(tooltips=[('Name', '@index')]))

# # Set layout
# # layout = column(slider,plot)
# layout = column(plot)

# # does not work
# #graph.node_renderer.data_source.on_change("selected", selected_points)

# # Create Bokeh server object
# curdoc().add_root(layout)
# # update()


# from bokeh.io import show, output_notebook
# from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool
# from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
# from bokeh.palettes import Spectral4
# from bokeh.models import LabelSet

# plot = Plot(plot_width=900, plot_height=500,
#             x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
# plot.title.text = "Graph Interaction Demonstration"

# plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())

# graph_renderer = from_networkx(G, nx.circular_layout, scale=1, center=(0,0))

# graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
# graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
# graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])
# graph_renderer.node_renderer.glyph.properties_with_values()
# graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
# graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
# graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

# graph_renderer.selection_policy = NodesAndLinkedEdges()
# graph_renderer.inspection_policy = EdgesAndLinkedNodes()

# plot.renderers.append(graph_renderer)

# # show(plot)
# curdoc().add_root(plot)

# p = figure(x_range=(0, 10), y_range=(0, 10))

# cds = ColumnDataSource(data=dict(x_start=[0,1, 2], y_start=[0,1, 2], x_end=[1,3, 5], y_end=[0,5, 8]))
# p.add_layout(Arrow(end=OpenHead(), source=cds, x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end'))

# pn.Column(p)

import networkx as nx

G = nx.Graph()
G.add_edge(1,2,color='r',weight=2)
G.add_edge(2,3,color='b',weight=4)
G.add_edge(3,4,color='g',weight=6)

pos = nx.circular_layout(G)

edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
weights = [G[u][v]['weight'] for u,v in edges]

nx.draw(G, pos, edges=edges, edge_color=colors, width=weights)