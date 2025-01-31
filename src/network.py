# Network modeling (Cell2Fire)
import os, sys
from shapely.geometry import Point, LineString
sys.path.append('../')
from utils import *

import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
import tqdm
import numpy as np
import pandas as pd
from rasterio.io import MemoryFile
from rasterio.mask import mask

# Global variables
epsg_utm = "EPSG:25831"
epsg_latlon = "EPSG:4326"

def generate_network(fuels=None, 
                    nsims=1, 
                    viz=False, 
                    src=None, 
                    fuels_fname='fuel_clipped.asc',
                    c2f_results_path=None, 
                    attribute="ros",
                    base_path=None, 
                    get_nodes=False,
                    lut_dictionary_path='/home/minho/research/fire/networks/C2F-W/spain_lookup_table.csv' # Set this path to where LUT is
                    ):

    # Set Messages path (From Cell2Fire results)
    _MessagesPath = os.path.join(c2f_results_path, 'Messages')

    _Rows, _Cols = fuels.shape
    _NCells = _Rows * _Cols

    # Load LUTs
    FBPDict, ColorsDict = Dictionary(lut_dictionary_path)
    Colors = ColorsDict

    # Load fuel map as a grid
    GForestN, GForestType, Rows, Cols, AdjCells, CoordCells, CellSide = ForestGrid(os.path.join(base_path, fuels_fname), FBPDict)

    # Graph generation
    nodes = range(1, _NCells+1)

    # We build a Digraph (directed graph, general graph for the instance)
    _GGraph = nx.DiGraph()

    # We add nodes to the list
    _GGraph.add_nodes_from(nodes)
    # for n in nodes:
    for i, n in enumerate(nodes):
        # n = flip(_n, fuels_meta['height'], fuels_meta['width'])
        _GGraph.nodes[n][attribute] = 0
        _GGraph.nodes[n]["time"] = 0
        _GGraph.nodes[n]["count"] = 0
        
    _nDigits = len(str(nsims))

    for k in range(1, nsims + 1):
        msgFileName = f"MessagesFile{k:0{_nDigits}}"
        
        # msgFileName = "MessagesFile0" if (k < 10) else "MessagesFile"
        if nsims == 1:
            msgFileName = "MessagesFile1" 

        if len(pd.read_csv(os.path.join(_MessagesPath, msgFileName + '.csv')).columns) == 4:
            HGraphs = nx.read_edgelist(path=_MessagesPath + '/' + msgFileName + '.csv',
                                        create_using=nx.DiGraph(),
                                        nodetype=int,
                                        data=[('time', float), (attribute, float)],
                                        delimiter=',')
        else: 
            HGraphs = nx.read_edgelist(path=_MessagesPath + '/' + msgFileName + '.csv',
                                        create_using=nx.DiGraph(),
                                        nodetype=int,
                                        data=[('time', float), ('ros', float), ('fli', float), ('fl', float)],
                                        delimiter=',')

        for e in HGraphs.edges():

            if _GGraph.has_edge(*e):
                _GGraph.get_edge_data(e[0], e[1])["weight"] += 1
                _GGraph.nodes[e[1]][attribute] += HGraphs[e[0]][e[1]][attribute]
                _GGraph.nodes[e[1]]["time"] += HGraphs[e[0]][e[1]]["time"]
                _GGraph.nodes[e[1]]["count"] += 1
                _GGraph.add_edge(e[0], e[1], 
                                att=HGraphs[e[0]][e[1]][attribute], 
                                time=HGraphs[e[0]][e[1]]["time"],
                                cell1=HGraphs[e[0]][e[1]]["cell1"],
                                cell2=HGraphs[e[0]][e[1]]["cell2"])
            else:
                _GGraph.add_weighted_edges_from([(*e,1.)])
                _GGraph.add_edge(e[0], e[1], att=HGraphs[e[0]][e[1]][attribute], time=HGraphs[e[0]][e[1]]["time"])

    # Average attribute (e.g., ROS), time
    for n in nodes:
        if _GGraph.nodes[n]['count'] > 0:
            _GGraph.nodes[n][attribute] /= _GGraph.nodes[n]['count']
            _GGraph.nodes[n]['time'] /= _GGraph.nodes[n]['count']
        else:
            _GGraph.nodes[n][attribute] = np.nan
            _GGraph.nodes[n]['time'] = np.nan

    # Re-orient the graph (Key step!!!)
    oa = CoordCells[::-1]

    coordinates = tuple(map(tuple, CoordCells))
    reshaped_array = np.array(coordinates).reshape(_Rows, _Cols, 2)

    coordinates_flipped = tuple(map(tuple, CoordCells[::-1]))
    reshaped_array_flipped = np.array(coordinates_flipped).reshape(_Rows, _Cols, 2)

    comb_array = np.dstack((reshaped_array[:,:,0], reshaped_array_flipped[:,:,1]))
    coordinates_new = comb_array.reshape(_Rows * _Cols, 2)

    # Set coordinate positions
    coord_pos = dict() # Cartesian coordinates

    for i in _GGraph.nodes:
        coord_pos[i] = coordinates_new[i-1] + 0.5

    # Create a list to store LineString objects representing edges
    edge_lines = []
    time_list, att_list, source_list, target_list = [],[],[],[]

    # Iterate over edges and convert them to LineString objects
    for start_id, end_id, weight_dict in tqdm.tqdm(_GGraph.edges(data=True), total=len(_GGraph.edges(data=True)), desc="Creating edges"):

        start_node = rowcol_to_geo(src, coord_pos[start_id])
        end_node = rowcol_to_geo(src, coord_pos[end_id])
        time = weight_dict['time']
        att_var = weight_dict['att']

        edge_lines.append(LineString([start_node,end_node]))
        time_list.append(time)
        att_list.append(att_var)
        source_list.append(start_id)
        target_list.append(end_id)

    # Create a GeoDataFrame from the LineString objects
    edges_gdf = gpd.GeoDataFrame(geometry=edge_lines)
    edges_gdf['time'] = time_list
    edges_gdf[attribute] = att_list
    edges_gdf['source'] = source_list
    edges_gdf['target'] = target_list

    # Create a GeoDataFrame for nodes including attributes
    node_list, source_list = [], []

    for start_id, weight_dict in tqdm.tqdm(_GGraph.nodes(data=True), total=len(_GGraph.nodes(data=True)), desc="Creating nodes"):
        start_node = rowcol_to_geo(src, coord_pos[start_id])

        node_list.append(Point([start_node]))
        source_list.append(start_id)

    # Create a GeoDataFrame from the Point objects
    nodes_gdf = gpd.GeoDataFrame(geometry=node_list)
    nodes_gdf['source'] = source_list

    # Map to nodes
    edges_time = edges_gdf.groupby('source')['time'].min().to_dict()
    edges_att = edges_gdf.groupby('source')[attribute].mean().to_dict()
    nodes_gdf['time'] = nodes_gdf['source'].map(edges_time)
    nodes_gdf[attribute] = nodes_gdf['source'].map(edges_att)

    # Filter NaN values
    nodes_gdf_filt = nodes_gdf.dropna(subset=['time', attribute]).reset_index(drop=True)
    
    # Remove nodes with NaN attributes from _GGraph
    filtered_node_ids = nodes_gdf_filt['source'].tolist()
    nodes_to_remove = [n for n in _GGraph.nodes if n not in filtered_node_ids]
    _GGraph_nodes = _GGraph.copy()
    _GGraph_nodes.remove_nodes_from(nodes_to_remove)

    # if viz:
    #     fig, ax = plt.subplots(1, 2, figsize=(15,5))

    #     polys_gdf.set_crs(epsg_utm).to_crs(epsg_latlon).plot(ax=ax[0], color='grey', edgecolor='k', alpha=0.5)
    #     edges_gdf_save.plot(ax=ax[0], column='time', cmap='Blues', legend=True)
    #     ax[0].set_title("Arrival Time [min]")

    #     polys_gdf.set_crs(epsg_utm).to_crs(epsg_latlon).plot(ax=ax[1], color='grey', edgecolor='k', alpha=0.5)
    #     edges_gdf_save.plot(ax=ax[1], column=attribute, cmap='Reds', legend=True)
    #     ax[1].set_title("ROS [m/min]")

    #     plt.show()

    if get_nodes:
        return _GGraph, _GGraph_nodes, edges_gdf, nodes_gdf_filt
    else: 
        return _GGraph, edges_gdf


def rowcol_to_geo(src, geometry):
    transform = src.transform
    transformed_x = transform[0]*geometry[0] + transform[1]*geometry[1]
    transformed_y = transform[3]*geometry[0] + transform[4]*geometry[1]

    x = transformed_x + transform[2]
    y = transformed_y + transform[5]
    return x, y    

def compute_avg_edge_weights(polys_gdf=None, edges_gdf=None):
    
    # Create an empty DataFrame to store results
    results = []

    # Iterate over each multipolygon in Shapefile A
    for index, row in tqdm.tqdm(polys_gdf.iterrows(), total=len(polys_gdf)):
        multipolygon = row['geometry']
        intersected_lines = edges_gdf[edges_gdf.intersects(multipolygon)]
        
        # Calculate average time and ros
        if not intersected_lines.empty:
            avg_time = intersected_lines['time'].mean()
            avg_ros = intersected_lines['ros'].mean()
        else:
            avg_time = None
            avg_ros = None
        
        # Store results
        results.append({'geometry': multipolygon, 'avg_time': avg_time, 'avg_ros': avg_ros})

    # Create a new GeoDataFrame from results
    results_gdf = gpd.GeoDataFrame(results, geometry='geometry')

    return results_gdf  


def compute_polygon_components(intersected_lines, orig_poly_idx, centroid, shapefile_a, attribute='ros'):
    
    # Desintation node of LineString
    rr = {'polygon_id0':[], 'polygon_id1':[], attribute:[], 'time':[], 'dist_start':[], 'dist_end':[], 'geometry':[], 'source':[], 'target':[]}

    for _, row in intersected_lines.iterrows():
        
        # Actual coordinates
        n0 = Point(row['geometry'].coords[0])
        n1 = Point(row['geometry'].coords[1])

        # Find which polygon the node overlaps
        for idx, poly in shapefile_a.iterrows():
            
            multipolygon = poly['geometry'] 
            
            if multipolygon.contains(n1):
                new_idx = idx
                new_node = multipolygon.centroid
                dist_to_n0 = centroid.distance(n0) # Distance from centroid to start node
                dist_to_n1 = new_node.distance(n1) # Distance from centroid to end node (neighbor)
                att_val = row[attribute]
                time = row['time']                
        
                rr['polygon_id0'].append(orig_poly_idx)
                rr['polygon_id1'].append(new_idx)
                rr[attribute].append(att_val)
                rr['time'].append(time)
                rr['dist_start'].append(dist_to_n0)
                rr['dist_end'].append(dist_to_n1)
                rr['geometry'].append(row['geometry'])
                rr['source'].append(row['source'])
                rr['target'].append(row['target'])
    
    return pd.DataFrame(rr)
    

def compute_intersecting_edges(polys_gdf=None, edges_gdf=None, attribute='ros'):

    # Create dictionary of centroids from polygons
    polygon_centroid_dict = {index: poly['geometry'].centroid for index, poly in polys_gdf.iterrows()}

    gg = []

    # Iterate over each multipolygon in Shapefile A
    for index_a, row_a in tqdm.tqdm(polys_gdf.iterrows(), total=len(polys_gdf)):
        multipolygon_a = row_a['geometry']
        centroid = multipolygon_a.centroid
        
        # Find LineStrings that intersect with multipolygon_a
        intersected_lines = edges_gdf[edges_gdf.intersects(multipolygon_a)]        
            
        if not intersected_lines.empty:
            
            # Find the destination node of the connected polygons for each LineString
            intersected_results = compute_polygon_components(intersected_lines, index_a, centroid, polys_gdf, attribute=attribute)
            intersected_results['start_node'] = intersected_results['polygon_id0'].map(polygon_centroid_dict) # Add end nodes from polygon IDs
            intersected_results['end_node'] = intersected_results['polygon_id1'].map(polygon_centroid_dict) # Add end nodes from polygon IDs

            gg.append(intersected_results)  
        
    final_df = pd.concat(gg, ignore_index=True)
    final_gdf = gpd.GeoDataFrame(final_df, geometry='geometry')
    return final_gdf


def create_suppression_network(final_df=None, polys_gdf=None, directed=False, save=False, attribute='ros',
                                sdi_slp=None, metadata=None, # Needed for PW-ROS
                                results_path=None,
                                network_path=None):

    # Node weights (Same polygons) --> only need polygon_id0
    node_final_gdf = final_df[final_df['polygon_id0'] == final_df['polygon_id1']]
    node_results = final_df[['polygon_id0',attribute,'time','dist_start','dist_end']].groupby('polygon_id0').mean().reset_index()
    min_time = final_df[['polygon_id0',attribute,'time','dist_start','dist_end']].groupby('polygon_id0').min().reset_index()['time'] # Get minimum time
    node_results['time'] = min_time

    # Edge weights (Diff polygons)
    edge_final_gdf = final_df[final_df['polygon_id0'] != final_df['polygon_id1']]

    ##### SDI & Time weighted metric
    if len(sdi_slp)>1:
        
        # Create an in-memory raster
        with MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                width=metadata['width'],
                height=metadata['height'],
                count=metadata['count'],
                dtype=metadata['dtype'],
                crs=metadata['crs'],
                transform=metadata['transform'],
            ) as src:
                src.write(sdi_slp, 1)  # Write the data to the first band

                results = []

                try:
                    # Iterate over each polygon
                    for idx, row in edge_final_gdf.iterrows():
                        geom = [row["geometry"]]  # Geometry as a list for rasterio.mask

                        # try:
                        # Mask the raster with the polygon geometry
                        masked_image, masked_transform = rasterio.mask.mask(src, geom, crop=True)

                        # Flatten the raster values and remove NoData
                        raster_values = masked_image[0].flatten()
                        raster_values = raster_values[raster_values != src.nodata]

                        # Aggregate raster values
                        if raster_values.size > 0:
                            mean_value = np.nanmean(raster_values)
                            sum_value = np.nansum(raster_values)
                            max_value = np.nanmax(raster_values)
                            min_value = np.nanmin(raster_values)
                        else:
                            mean_value, sum_value, max_value, min_value = np.nan, np.nan, np.nan, np.nan

                        # Append results
                        results.append({
                            "polygon_id": idx,          # Use index or a unique column from roads_bnds
                            "sdi_mean": mean_value,
                            "sdi_sum": sum_value,
                            "sdi_max": max_value,
                            "sdi_min": max_value,
                        })

                except Exception as e:
                    print(f"Error processing polygon {idx}: {e}")

            # Convert results to a DataFrame and merge with GeoDataFrame
            edge_results_gdf = pd.DataFrame(results)

            # # Ensure index alignment for merging
            edge_final_gdf = final_df.reset_index().merge(edge_results_gdf.reset_index(), left_on="index", right_on="polygon_id")

    # Create common key for group
    edge_final_gdf['comb'] = edge_final_gdf['polygon_id0'].astype(str) + '_' + edge_final_gdf['polygon_id1'].astype(str)

    # Group
    main_list = ['polygon_id0','polygon_id1',attribute,'time','dist_start','dist_end','comb']

    if len(sdi_slp)>1:
        main_list += ['sdi_mean', 'sdi_sum', 'sdi_max', 'sdi_min']
        attribute = "PW-ROS"
        
    edge_results = edge_final_gdf[main_list].groupby('comb').mean().reset_index()

    # Sort
    edge_results = edge_results.sort_values(by='polygon_id0')    

    # Create dictionary of centroids from polygons
    polygon_centroid_dict = {index: poly['geometry'].centroid for index, poly in polys_gdf.iterrows()}


    # Compute Time & SDI weighted Penetration ROS (PW-ROS)
    # Penetration ROS = (\Sigma(ROS_i * W_i)) / \Sigma(W_i)
    # where W_i = (SDI / t_i) * vector density
    if len(sdi_slp)>1:
        list_edges, list_ed = [], []

        for idx, row in edge_results.iterrows():
            
            edges = edge_final_gdf[edge_final_gdf['comb'] == row['comb']]
            edges_buffered_gdf = gpd.GeoDataFrame(geometry=edges.buffer(10)).set_crs(epsg_utm)
            
            # Compute polygon boundary
            poly0_gdf = gpd.GeoDataFrame(geometry=polys_gdf[polys_gdf.index==row['polygon_id0']].geometry).set_crs(epsg_utm)
            poly1_gdf = gpd.GeoDataFrame(geometry=polys_gdf[polys_gdf.index==row['polygon_id1']].geometry).set_crs(epsg_utm)
            overlap_bnd_length = poly0_gdf.clip(poly1_gdf).length.sum()

            overlap_edges_length = edges.clip(edges_buffered_gdf).length.sum()     # Length of overlapping edges
            edge_density = overlap_edges_length / overlap_bnd_length               # Fraction of overlap

            # Compute weight for each edge
            edges['w'] = edge_density
            
            # Compute Time and SDI weighted penetration ROS
            edges['PW-ROS'] = edges['ros'] * edges['w']
            
            list_edges.append(edges)
            list_ed.append(edge_density)

        # Merge the DataFrames and populate PW-ROS
        updated_gdf = edge_final_gdf.merge(
            pd.concat(*[list_edges])[main_list + ['PW-ROS']], 
            on=main_list, 
            how='left'
        )

        updated_gdf['n_PW-ROS'] = updated_gdf['PW-ROS'] / (updated_gdf['PW-ROS'].sum()) 
        edge_results = updated_gdf[main_list + ['PW-ROS']].groupby('comb').max().reset_index()
        
        edge_results['edge_density'] = list_ed
        ed_min = np.nanmin(list_ed)
        ed_max = np.nanmax(list_ed)
        list_ed_norm=[]
        for i, d in edge_results.iterrows():
            ed_norm = (d['edge_density'] - ed_min) / (ed_min + ed_max)
            list_ed_norm.append(ed_norm)    
        edge_results['edge_density'] = list_ed_norm

        edge_results = edge_results.sort_values(by='polygon_id0')    

    ## Network
    network = nx.Graph()

    if directed:
        network = nx.DiGraph()

    ### Node parameters
    for idx, _node in node_results.iterrows():

        node = int(_node['polygon_id0'])
        position = (polygon_centroid_dict[_node['polygon_id0']].x, polygon_centroid_dict[_node['polygon_id0']].y)
        node_ros = _node['ros']
        node_time = _node['time']

        network.add_node(node, pos=position, att=node_ros, time=node_time)

    ### Edge parameters
    for idx, _edge in edge_results.iterrows():
        node0 = _edge['polygon_id0']
        node1 = _edge['polygon_id1']
        edge_ros = _edge[attribute]
        edge_time = _edge['time']

        network.add_edge(node0, node1, att=edge_ros, time=edge_time)    

        if len(sdi_slp) > 1:
            
            network.add_edge(node0, node1, ed=_edge['edge_density'], time=edge_time)    


    if save:
        edges_gdf_save = edge_results.set_crs(epsg_utm).to_crs(epsg_latlon)
        polys_gdf_save = polys_gdf.set_crs(epsg_utm).to_crs(epsg_latlon)
        
        node_results.to_csv(os.path.join(results_path, network_path + '_nodes.csv'))
        edge_results.to_csv(os.path.join(results_path, network_path + '_edges.csv'))
        edges_gdf_save.to_file(os.path.join(results_path, network_path + '_edges_gdf.shp'))
        polys_gdf_save.to_file(os.path.join(results_path, network_path + '_polys_gdf.shp'))

    return network


def model_network(fuels, nsims, fuels_src, fuels_fname, base_path, c2f_results_path, polys_gdf, directed_option, save_option, attribute='ros'):

    # Create directed network of weighted, propagation tree and vectorized gdf
    '''
    - main_graph: Directed network of weighted, propagation tree
    - edges_gdf: Vectorized line strings of network edges with weights as individual columns. 
                Also includes source and target nodes (used later in tactical interventions)
    '''

    main_graph, edges_gdf = generate_network(fuels=fuels, nsims=nsims, src=fuels_src, fuels_fname=fuels_fname, attribute=attribute,
                                            c2f_results_path=c2f_results_path, base_path=base_path)

    # Compute average weights (i.e., Avg. ROS and avg. time) per polygon (outputted as GeoPandas GDF)
    # results_gdf = compute_avg_edge_weights(polys_gdf, edges_gdf)

    # Compute average edge weights per network edge
    final_gdf = compute_intersecting_edges(polys_gdf, edges_gdf)
    
    # Create suppression network
    network = create_suppression_network(final_df=final_gdf, polys_gdf=polys_gdf, directed=directed_option, save=save_option)

    return main_graph, edges_gdf, final_gdf, network

#TODO: SPEED UP!

