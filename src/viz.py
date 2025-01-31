# Visualization
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
import networkx as nx
import contextily as ctx
import numpy as np
import pandas as pd
import rasterio
import tqdm
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point

from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe

CRS_BASE = 25831 # Catalonia

def viz_network_v2(network, polys_gdf, figsize=(15,11), crs=CRS_BASE, edges_gdf=None, roads_gdf=None, directed=True, save=False, output_path=None, boundaries='time', attribute='ros', attribute_label="ROS [m/min]\non edges", complexity_gdf=None):
    
    # Colormaps for visualization
    cmap_connections = sns.color_palette("RdYlGn_r", as_cmap=True)      # Connections
    cmap_time = plt.get_cmap('GnBu')                                    # Time (Nodes and boundaries)

    # Create network
    network = nx.DiGraph(network) if directed else nx.Graph(network)

    ### Plot
    fig, ax = plt.subplots(1, figsize=figsize)

    # Plot 1: Roads with legend
    roads_gdf.plot(ax=ax, color='k', linewidths=0.5, alpha=0.5, label='Roads') if roads_gdf is not None else None
    # plt.legend()

    # Plot 2: Main polygons
    polys_gdf.plot(ax=ax, color='none', linewidths=1.5, edgecolor='k')

    # Plot 2-1: Draw nodes
    node_positions = nx.get_node_attributes(network, 'pos')

    nx.draw_networkx_nodes(network, pos=node_positions, node_size=1000, 
                            cmap=cmap_time, node_color=list(nx.get_node_attributes(network, 'time').values()),
                            alpha=0.8, edgecolors='black', linewidths=1, label=True,
                            ax=ax)
    nx.draw_networkx_labels(network, pos=node_positions, font_size=24)

    # Plot 2-2: Draw edges
    max_ros = max(list(nx.get_edge_attributes(network, attribute).values()))
    # weights = [10 * (network[u][v]['ros'] / max_ros) for u, v in network.edges()]
    nx.draw_networkx_edges(network, pos=node_positions, 
                            width=10,
                            edge_color=list(nx.get_edge_attributes(network, attribute).values()),
                            edge_cmap=cmap_connections,
                            connectionstyle='arc3,rad=0.3', 
                            arrowsize=20,
                            alpha=0.75,
                            ax=ax)

    # Visualization details
    # Add ROS color bar
    cax1 = fig.add_axes([0.9, 0.5, 0.02, 0.25])  # [left, bottom, width, height]
    norm1 = Normalize(vmin=0, vmax=max(list(nx.get_edge_attributes(network, attribute).values())))
    scalar_map1 = ScalarMappable(cmap=cmap_connections, norm=norm1)
    scalar_map1.set_array(list(nx.get_edge_attributes(network, attribute).values()))
    cbar1 = fig.colorbar(scalar_map1, cax=cax1)
    cbar1.set_label(attribute_label, size=24)
    cbar1.ax.tick_params(labelsize=24)  # Increase font size for colorbar ticks

    # Add Arrival Time color bar
    cax2 = fig.add_axes([0.9, 0.2, 0.02, 0.25])  # [left, bottom, width, height]
    norm2 = Normalize(vmin=0, vmax=max(list(nx.get_node_attributes(network, 'time').values())))
    scalar_map2 = ScalarMappable(cmap=cmap_time, norm=norm2)
    scalar_map2.set_array(list(nx.get_node_attributes(network, 'time').values()))
    cbar2 = fig.colorbar(scalar_map2, cax=cax2)
    cbar2.set_label('Elapsed Time [min]\non nodes', size=24)
    cbar2.ax.tick_params(labelsize=24)  # Increase font size for colorbar ticks

    ctx.add_basemap(ax=ax, crs=crs, source=ctx.providers.CartoDB.Positron)
    ax.add_artist(ScaleBar(dx=1, label="Scale", location="lower right", label_loc="top", scale_loc="bottom"))

    # Set extent boundaries
    xmin, ymin, xmax, ymax = polys_gdf.buffer(0.001).total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=16)

    plt.show()

def viz_network_with_roads(network, polys_gdf, edges_gdf=None, roads_gdf=None, directed=True, save=False, output_path=None, boundaries='time', complexity_gdf=None):

    # Colormaps for visualization
    cmap_connections = sns.color_palette("RdYlGn_r", as_cmap=True)      # Connections
    cmap_time = plt.get_cmap('coolwarm_r')                              # Time (Nodes and boundaries)
    cmap_fire = plt.get_cmap('gist_heat_r')                             # Boundaries (optional)

    # Create network
    network = nx.DiGraph(network) if directed else nx.Graph(network)
    
    ### Plot
    fig, ax = plt.subplots(1, figsize=(15, 12))

    # Plot 1: Roads with legend
    roads_gdf.plot(ax=ax, color='k', linewidths=0.5, alpha=0.5, label='Roads') if roads_gdf is not None else None
    # plt.legend()

    # Plot 2: Main polygons
    polys_gdf.plot(ax=ax, color='none', linewidths=1.5, edgecolor='k')

    # Plot 2-1: Draw nodes
    node_positions = nx.get_node_attributes(network, 'pos')

    nx.draw_networkx_nodes(network, pos=node_positions, node_size=200, 
                           cmap=cmap_time, node_color=list(nx.get_node_attributes(network, 'time').values()),
                           alpha=0.8, edgecolors='black', linewidths=1, label=True,
                           ax=ax)
    nx.draw_networkx_labels(network, pos=node_positions, font_size=10)

    # Plot 2-2: Draw edges
    max_ros = max(list(nx.get_edge_attributes(network, 'ros').values()))
    weights = [10 * (network[u][v]['ros'] / max_ros) for u, v in network.edges()]
    nx.draw_networkx_edges(network, pos=node_positions, 
                           width=3,
                           edge_color=list(nx.get_edge_attributes(network, 'ros').values()),
                           edge_cmap=cmap_connections,
                           connectionstyle='arc3,rad=0.3', 
                           arrowsize=20,
                           alpha=0.75,
                           ax=ax)

    xmin, ymin, xmax, ymax = polys_gdf.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Visualization details
    # Add ROS color bar
    cax1 = fig.add_axes([0.85, 0.7, 0.02, 0.25])  # [left, bottom, width, height]
    norm1 = Normalize(vmin=0, vmax=max(list(nx.get_edge_attributes(network, 'ros').values())))
    scalar_map1 = ScalarMappable(cmap=cmap_connections, norm=norm1)
    scalar_map1.set_array(list(nx.get_edge_attributes(network, 'ros').values()))
    cbar1 = fig.colorbar(scalar_map1, cax=cax1)
    cbar1.set_label('ROS [m/min] on edges')

    # Add Arrival Time color bar
    cax2 = fig.add_axes([0.85, 0.4, 0.02, 0.25])  # [left, bottom, width, height]
    norm2 = Normalize(vmin=0, vmax=max(list(nx.get_node_attributes(network, 'time').values())))
    scalar_map2 = ScalarMappable(cmap=cmap_time, norm=norm2)
    scalar_map2.set_array(list(nx.get_node_attributes(network, 'time').values()))
    cbar2 = fig.colorbar(scalar_map2, cax=cax2)
    cbar2.set_label('Arrival Time [min] on nodes')

    # Add color based on attribute on polygon boundaries (Optional)
    if boundaries:

        def polygon_edges_and_centers(polygon):
            exterior_coords = list(polygon.exterior.coords)
            edges_centers = []
            for i in range(len(exterior_coords) - 1):
                edge = LineString([exterior_coords[i], exterior_coords[i + 1]])
                center_point = edge.interpolate(0.5, normalized=True)
                edges_centers.append((edge, center_point))
            return edges_centers

        centers_list = []

        # Process each basin to find its edges and compute ROS at center points
        for idx, basin in polys_gdf.iterrows():
            edges_centers = polygon_edges_and_centers(basin['geometry'])
            for edge, center_point in edges_centers:
                nearest_edge = edges_gdf.iloc[edges_gdf.distance(center_point).idxmin()]
                ros_value = nearest_edge['ros']
                time_value = nearest_edge['time']
                centers_list.append({'geometry': center_point, 'ros': ros_value, 'time': time_value})

        centers_with_values = gpd.GeoDataFrame(centers_list, crs=polys_gdf.crs)
        centers_with_values['color'] = centers_with_values[boundaries].apply(lambda x: cmap_time(norm2(x)))
        # centers_with_values.plot(ax=ax, color=centers_with_values['color'], vmin=0, markersize=20, alpha=0.25, zorder=1)
        centers_with_values.plot(ax=ax, column='time', cmap=cmap_time, vmin=0, markersize=20, alpha=0.5, zorder=1)

    # Plot fire path complexity
    complexity_gdf.plot(ax=ax, cmap=cmap_fire, column='count_ros', alpha=0.3, zorder=0)
    norm = Normalize(vmin=0, vmax=complexity_gdf['count_ros'].max())

    # Add boundary color bar
    cax3 = fig.add_axes([0.85, 0.1, 0.02, 0.25])  # [left, bottom, width, height]
    sm = ScalarMappable(cmap=cmap_fire, norm=norm)
    sm.set_array([])
    cbar3 = fig.colorbar(sm, cax=cax3)
    cbar3.set_label('Fire path complexity')

    if save:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

    # return 


def plot_zoom(superpixel_gdf, final_gdf, roads_gdf, output_path=None, save=False):
    # Create bbox around merged superpixels for clipping
    bbox=merged_superpixel_gdf.total_bounds
    bbox_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), 
                            (bbox[2], bbox[3]), (bbox[0], bbox[3])])
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_polygon])

    # Find ROS vectors that traverse two different superpixels (i.e., critical ROS vectors)
    edge_final_gdf = final_gdf[final_gdf['superpixel_id0'] != final_gdf['superpixel_id1']]

    # Create LineString geometries from start and end points
    edge_final_gdf['geometry'] = edge_final_gdf.apply(lambda row: shapely.geometry.LineString([row['start_node'], row['end_node']]), axis=1)

    int_edges_plot = final_gdf[(final_gdf['superpixel_id0'] == superpixel_ids[0]) & (final_gdf['superpixel_id1'] == superpixel_ids[1])]
    roads_clipped = roads_gdf.clip(bbox_gdf) # Roads clipped

    fig, ax = plt.subplots(1,2)

    superpixel_list.plot(ax=ax[0], color='none', edgecolor='k')
    int_edges_plot.plot(ax=ax[0], cmap='Reds', linewidth=5)    
    roads_clipped.plot(ax=ax[0], color='blue', linewidth=0.5, label='Roads', legend=True)

    int_edges_plot.plot(ax=ax[1], color='red', linewidth=1)

    plt.tight_layout()
    if save:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_zoom_v2(final_supp_edges, crs, polys_gdf, roads_gdf, final_gdf, superpixel_ids, output_path=None, save=None):
    unary_edges = gpd.GeoDataFrame(geometry=[final_supp_edges.unary_union], crs=crs)
    bbox = unary_edges.total_bounds
    bbox_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), 
                            (bbox[2], bbox[3]), (bbox[0], bbox[3])])
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_polygon]).buffer(200)

    # # Plot remaining (not suppressed)
    # remaining = final_gdf[final_gdf['superpixel_id0'] != final_gdf['superpixel_id1']]
    # remaining_gdf = remaining[(remaining['superpixel_id0'] == superpixel_ids[0]) & (remaining['superpixel_id1'] == superpixel_ids[1])]

    # # Plot separately
    sup1 = gpd.GeoDataFrame([polys_gdf.loc[superpixel_ids[0]]]).clip(bbox_gdf)
    sup2 = gpd.GeoDataFrame([polys_gdf.loc[superpixel_ids[1]]]).clip(bbox_gdf)

    fig, ax = plt.subplots(1, figsize=(12,12))
    roads_gdf.clip(bbox_gdf).plot(ax=ax, color='k', label='Road')
    bbox_gdf.plot(ax=ax, color='none', linestyle='--')
    
    final_supp_edges.plot(ax=ax, color='red', linewidth=2, label='Suppressed edges')
    final_supp_edges = final_supp_edges[final_supp_edges['geo'].apply(lambda x: isinstance(x, LineString))]
    for idx, row in tqdm.tqdm(final_supp_edges.iterrows(), total=len(final_supp_edges)):
        coords = row['geo'].coords
        end_point = coords[-1]
        ax.annotate("", xy=end_point, xytext=coords[-2], arrowprops=dict(arrowstyle="->", color='r', mutation_scale=10))

    # remaining_gdf.plot(ax=ax, color='yellow', label='Lost edges')
    # final_gdf.clip(bbox_gdf).plot(ax=ax, color='grey', alpha=0.25)

    bg = final_gdf.clip(bbox_gdf)
    bg = bg[bg['geo'].apply(lambda x: isinstance(x, LineString))]

    for idx, row in tqdm.tqdm(bg.iterrows(), total=len(bg)):
        # Extract coordinates of the linestring
        coords = row['geo'].coords
        # Get the coordinates of the end point
        end_point = coords[-1]
        # Plot arrow at the end of the linestring
        ax.annotate("", xy=end_point, xytext=coords[-2], arrowprops=dict(arrowstyle="->", color='grey', alpha=0.25, mutation_scale=10))

    sup1.plot(ax=ax, color='grey', alpha=0.5, label='Superpixel'+str(superpixel_ids[0]), legend=True)
    sup2.plot(ax=ax, color='royalblue', alpha=0.5, label='Superpixel'+str(superpixel_ids[1]), legend=True)

    plt.legend()

    if save:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

    return final_supp_edges

def plot_fuelmaps(fuel_path=None, suppressed_fuel_path=None, output_path=None, nrows=788, ncols=783, polys_gdf_cleaned=None, tactic_path=None, save_option=False, meta=None):
    
    # 0. Fuel map parameters
    df = pd.read_csv(fuel_path)
    grid = df['ftypeN'].values.reshape(nrows, ncols)

    ascii_grid = np.loadtxt(suppressed_fuel_path, skiprows=6)
    diff = grid != ascii_grid

    # 0. Plot
    
    # Get upper left, upper right, lower left, lower right coordinates
    upper_left = (meta['transform'][2], meta['transform'][5])
    upper_right = (meta['transform'][2] + meta['transform'][0] * meta['width'], meta['transform'][5])
    lower_left = (meta['transform'][2], meta['transform'][5] + meta['transform'][4] * meta['height'])
    lower_right = (meta['transform'][2] + meta['transform'][0] * meta['width'], meta['transform'][5] + meta['transform'][4] * meta['height'])

    # Extract X and Y values
    x_values = [upper_left[0], upper_right[0], lower_left[0], lower_right[0]]
    y_values = [upper_left[1], upper_right[1], lower_left[1], lower_right[1]]

    # Find min and max X and Y values
    total_extent = [np.nanmin(x_values), 
                    np.nanmax(x_values),
                    np.nanmin(y_values), 
                    np.nanmax(y_values),
                    ]

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15,5), sharex=True, sharey=True)

    ax[0].imshow(ascii_grid, extent=total_extent)
    ax[0].set_title('Original Fuel Map')
    ax[1].imshow(grid, extent=total_extent)
    ax[1].set_title('Suppressed Fuel Map')

    im=ax[2].imshow(grid != ascii_grid, cmap='Reds', extent=total_extent)
    ax[2].set_title('Difference')
    # fig.colorbar(im)

    for i in range(2):
        polys_gdf_cleaned.plot(ax=ax[i], color='none', edgecolor='k', linewidth=0.25)
        polys_gdf_cleaned.loc[tactic_path].plot(ax=ax[i], color='r', alpha=0.5)
    polys_gdf_cleaned.loc[tactic_path].plot(ax=ax[2], color='none', edgecolor='k', linewidth=0.25, alpha=0.5)

    plt.tight_layout()
    plt.show()

    if save_option:
        
        grid = grid.astype(np.int16)
        meta.update({
            'count': 1,  # number of bands
            'dtype': 'int16'
            })

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(grid, 1)