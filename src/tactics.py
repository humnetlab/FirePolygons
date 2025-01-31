# Tactical
import rasterio
import shapely
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import networkx as nx
from networkx import betweenness_centrality, pagerank
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def network_characteristics(network):

    bc = betweenness_centrality(network)
    deg = network.degree
    pr = pagerank(network)

    # Extract node coordinates and attributes
    nodes_data = []
    for node, attrs in network.nodes(data=True):
        node_data = {'node_id': node}
        node_data.update(attrs)  # Add other node attributes if any
        nodes_data.append(node_data)

    # Create a GeoDataFrame from node data
    gdf = gpd.GeoDataFrame(nodes_data)  # Assuming WGS84 coordinate reference system
    gdf['degree'] = [i[1] for i in list(deg)]
    gdf['bc'] = [i[1] for i in list(bc.items())]
    gdf['pr'] = [i[1] for i in list(pr.items())]

    gdf = gdf.drop(['geometry'], axis=1)

    return gdf

# Tactic decision-making utility functions
def calculate_weighted_sum(graph, path, weight=None):
    weighted_sum = 0
    for u, v in zip(path[:-1], path[1:]):
        weighted_sum += graph[u][v][weight]  
    return weighted_sum

def limited_shortest_paths_with_weighted_sum(graph, source, depth_limit):
    paths = nx.single_source_shortest_path(graph, source, cutoff=depth_limit)
    paths_with_weighted_sum = [(path, calculate_weighted_sum(graph, path, weight='ros'), calculate_weighted_sum(graph, path, weight='time')) for path in paths.values()]

    return paths_with_weighted_sum

# Tactic decision-making main function
def network_decision(network, polys_gdf, source_node=None, depth_limit=6, num_decisions=1, viz=False):
    
    # Set the source node and depth limit
    if depth_limit==None:
        depth_limit = network.degree(source_node)

    # Find all paths within the degree limit of the source node with their weighted sums
    paths_with_weighted_sum = limited_shortest_paths_with_weighted_sum(network, source_node, depth_limit)
    df_paths = pd.DataFrame(paths_with_weighted_sum, columns=['paths', 'sum_ros', 'time'])

    # Plot the top 3 paths using their actual coordinates
    if num_decisions > len(df_paths):
        print("Requested number of decisions is greater than the number of possible paths")
        
    top_paths = sorted(paths_with_weighted_sum, key=lambda x: x[1], reverse=True)[:num_decisions]

    if viz:
        for i, (path, weighted_sum, weighted_time) in enumerate(top_paths):

            fig, ax = plt.subplots(1, figsize=(12,12))
            polys_gdf.plot(ax=ax, color='none', linewidths=.5, edgecolor='k')

            ax.set_title("Top Path {:.0f} - Cumulative ROS: {:.3f} [m/min] and Time {:.3f} [min]".format(i+1, weighted_sum, weighted_time))

            # Draw nodes
            node_positions = nx.get_node_attributes(network, 'pos')
            # nx.draw_networkx_nodes(network, pos=node_positions, node_size=200, node_color='none',
            #                 alpha=0.5, edgecolors='black', linewidths=1, label=True,
            #                 ax=ax)
            node_positions = nx.get_node_attributes(network.subgraph(path), 'pos')
            nx.draw_networkx_nodes(network.subgraph(path), pos=node_positions, node_size=500, 
                                cmap=plt.cm.Blues, node_color=list(nx.get_node_attributes(network.subgraph(path), 'time').values()),
                                edgecolors='black', linewidths=1, label=True,
                                ax=ax)
            nx.draw_networkx_labels(network.subgraph(path), pos=node_positions,font_size=10)

            # # Draw edges
            edge_cmap = plt.cm.autumn_r
            max_ros = max(list(nx.get_edge_attributes(network.subgraph(path), 'ros').values()))
            weights = [5*(network.subgraph(path)[u][v]['ros'] / max_ros) for u,v in network.subgraph(path).edges()]
            nx.draw_networkx_edges(network.subgraph(path), pos=node_positions, label=True,
                                    width=weights,
                                    edge_color=list(nx.get_edge_attributes(network.subgraph(path), 'ros').values()),
                                    edge_cmap=edge_cmap,
                                    alpha=0.75,
                                    connectionstyle="arc3,rad=0.15",
                                    ax=ax)
            edge_labels = nx.get_edge_attributes(network.subgraph(path), "ros")
            edge_labels_r = {key: round(value, 1) for key, value in edge_labels.items()} # round to 1 dec place
            # nx.draw_networkx_edge_labels(network.subgraph(path), pos=node_positions, font_size=8, edge_labels=edge_labels_r, alpha=0.5, font_color='r', font_weight='bold')

            # Set x and y limits
            xmin, ymin, xmax, ymax = polys_gdf.total_bounds
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            norm = mcolors.Normalize(vmin=0,
                                    # vmin=min(list(nx.get_edge_attributes(network.subgraph(path), 'ros').values())), 
                                    vmax=max(list(nx.get_edge_attributes(network.subgraph(path), 'ros').values())))  # Setting vmax to 2 for gray scale
            scalar_map = plt.cm.ScalarMappable(cmap=edge_cmap)
            scalar_map.set_array(list(nx.get_edge_attributes(network.subgraph(path), 'ros').values()))
            cbar = plt.colorbar(scalar_map, shrink=0.5, ax=ax)
            cbar.set_label('ROS [m/min] on edges')

            # Create color bar
            cmap = plt.cm.Blues
            norm = mcolors.Normalize(vmin=0,
                                    # vmin=min(list(nx.get_node_attributes(network.subgraph(path), 'time').values())), 
                                    vmax=max(list(nx.get_node_attributes(network.subgraph(path), 'time').values())))
            scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

            scalar_map.set_array(list(nx.get_node_attributes(network.subgraph(path), 'time').values()))
            cbar = plt.colorbar(scalar_map, shrink=0.5, ax=ax)
            cbar.set_label('Arrival Time on nodes')

            plt.show()
    
    return df_paths, top_paths

def save_paths(paths_with_weighted_sums):
    
    # Initialize an empty list to store the rows of the DataFrame
    rows = []

    # Iterate over each key-value pair in the dictionary
    for key, values in paths_with_weighted_sum.items():
        # Iterate over each tuple in the list of values
        for value in values:
            # Append the key, tuple, and individual tuple elements to the rows list
            rows.append([key, *value])

    # Create a DataFrame from the rows list with appropriate column names
    df = pd.DataFrame(rows, columns=['source', 'path', 'sum_ros', 'sum_time'])
    
    return df