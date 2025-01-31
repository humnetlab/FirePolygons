# Network utils
import os, json, sys, shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
sys.path.append('../C2F-W')
import Cell2Fire.DataGeneratorC as DataGenerator
import rasterio
import rasterio.mask


def clip_raster_with_shp(input_array, raster_transform, raster_meta, shapefile, nodata=None, save_file=None):
    # Reproject shapefile to match the CRS of the raster image if necessary
    if shapefile.crs != raster_meta['crs']:
        shapefile = shapefile.to_crs(raster_meta['crs'])

    # Get features from the shapefile
    coords = get_features(shapefile)

    # Create an in-memory raster dataset using the input array
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=input_array.shape[0],
            width=input_array.shape[1],
            count=1,
            dtype=input_array.dtype,
            crs=raster_meta['crs'],
            transform=raster_transform,
        ) as dataset:
            dataset.write(input_array, 1)

            # Mask the raster image using the shapefile geometry
            clipped_array, clipped_transform = rasterio.mask.mask(dataset=dataset, shapes=coords, crop=True)

    # Update metadata
    clipped_meta = raster_meta.copy()
    clipped_meta.update({
        'transform': clipped_transform,
        'height': clipped_array.shape[1],
        'width': clipped_array.shape[2],
    })

    if nodata:
        clipped_array[clipped_array == nodata] = np.nan

    # Optionally save the clipped image to file
    if save_file:
        with rasterio.open(save_file, 'w', **clipped_meta) as dst:
            dst.write(clipped_array)

    return clipped_array[0], clipped_meta

def get_features(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][i]['geometry'] for i in range(len(gdf))]

def index_to_row_col(index, num_rows, num_columns):
    row = index // num_columns
    col = index % num_columns
    return row, col
    
#
def plt_style(self, b=True, l=True, r=False, t=False):
    '''
    https://matplotlib.org/stable/tutorials/introductory/customizing.html
    https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html
    , bbox_inches='tight', dpi=200
    '''
        #'backend':'QtAgg',
    params = { \
        'interactive': False,
        'font.size' : 16,
        'axes.labelsize' : 16,
        'axes.titlesize' : 16,
        'xtick.labelsize' : 16,
        'ytick.labelsize' : 16,
        'legend.fontsize' : 16,
        'figure.titlesize' : 18,
        'axes.spines.bottom': b,
        'axes.spines.left': l,
        'axes.spines.right': r,
        'axes.spines.top': t,
        'xtick.bottom': True,
        'ytick.left': True,
        'axes.facecolor':'white',
        'figure.facecolor':'white',
        }
    plt.rcParams.update( params)
    sns.set_theme( rc=params)
    sns.set_style( rc=params)
    sns.set_context( rc=params)
    sns.axes_style( rc=params)

# some stuff to make json work with python 2.7 taken from the web
def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data

#================

'''
Returns         dict {int: string}, dict {int: (double, double, double, double)}

Inputs:
filename        str
'''
# Reads fbp_lookup_table.csv and creates dictionaries for the fuel types and cells' colors
def Dictionary(filename):
    aux = 1
    file = open(filename, "r") 
    row = {}
    colors = {} 
    all = {}
    
    # Read file and save colors and ftypes dictionaries
    for line in file: 
        if aux > 1:
            aux +=1
            line = line.replace("-","")
            line = line.replace("\n","")
            line = line.replace("No","NF")
            line = line.split(",")
            
            if line[3][0:3] in ["FM1"]:
                row[line[0]] = line[3][0:4]
            elif line[3][0:3] in ["Non", "NFn"]:
                row[line[0]] = "NF"
            else:    
                row[line[0]] = line[3][0:3]
                
            colors[line[0]] = (float(line[4]) / 255.0, 
                               float(line[5]) / 255.0,
                               float(line[6]) / 255.0,
                               1.0)
            all[line[0]] = line
    
        if aux == 1:
            aux +=1
            
    return row, colors
    

'''
Returns         array of int, array of strings, int, int, dict {string:[int]}, int maxtrix NCells X 2, double

Inputs:
filename        str
Dictionary      Dictionary
'''   
# Reads the ASCII file with the forest grid structure and returns an array with all the cells and grid dimensions nxm
# Modified Feb 2018 by DLW to read the forest params (e.g. cell size) as well
def ForestGrid(filename, Dictionary):
    AdjCells = []
    North = "N"
    South = "S"
    East = "E"
    West = "W"
    NorthEast = "NE"
    NorthWest = "NW"
    SouthEast = "SE"
    SouthWest = "SW"
    
    with open(filename, "r") as f:
        filelines = f.readlines()

    line = filelines[4].replace("\n","")
    parts = line.split()
    
    if parts[0] != "cellsize":
        print ("line=",line)
        raise RuntimeError("Expected cellsize on line 5 of "+ filename)
    cellsize = float(parts[1])
    
    cells = 0
    row = 1
    trows = 0 
    tcols = 0
    gridcell1 = []
    gridcell2 = []
    gridcell3 = []
    gridcell4 = []
    grid = []
    grid2 = []
    
    # Read the ASCII file with the grid structure
    for row in range(6, len(filelines)):
        line = filelines[row]
        line = line.replace("\n","")
        line = ' '.join(line.split())
        line = line.split(" ")
        #print(line)
        
        
        for c in line: #range(0,len(line)-1):
            if c not in Dictionary.keys():
                gridcell1.append("NF")
                gridcell2.append("NF")
                gridcell3.append(int(0))
                gridcell4.append("NF")
            else:
                gridcell1.append(c)
                gridcell2.append(Dictionary[c])
                gridcell3.append(int(c))
                gridcell4.append(Dictionary[c])
            tcols = np.max([tcols,len(line)])

        grid.append(gridcell1)
        grid2.append(gridcell2)
        gridcell1 = []
        gridcell2 = []
    
    # Adjacent list of dictionaries and Cells coordinates
    CoordCells = np.empty([len(grid)*(tcols), 2]).astype(int)
    n = 1
    tcols += 1
    for r in range(0, len(grid)):
        for c in range(0, tcols - 1):
            #CoordCells.append([c,len(grid)-r-1])
            CoordCells[c + r*(tcols-1), 0] = c
            CoordCells[c + r*(tcols-1), 1] = len(grid)-r-1
            
            if len(grid) >1:
                
                if r == 0:
                    if c == 0:
                        AdjCells.append({North:None,NorthEast:None,NorthWest:None, 
                                         South:[n+tcols-1], SouthEast:[n+tcols], 
                                         SouthWest:None, East:[n+1],West:None})
                        n+=1
                    if c == tcols-2:
                        AdjCells.append({North:None,NorthEast:None,NorthWest:None,
                                         South:[n+tcols-1],SouthEast:None,SouthWest:[n+tcols-2], 
                                         East:None, West:[n-1]})
                        n+=1
                    if c>0 and c<tcols-2:    
                        AdjCells.append({North:None,NorthEast:None,NorthWest:None,
                                         South:[n+tcols-1],SouthEast:[n+tcols], 
                                         SouthWest:[n+tcols-2], East:[n+1],West:[n-1]})
                        n+=1
                
                if r > 0 and r < len(grid)-1:
                    if c == 0:
                        AdjCells.append({North:[n-tcols+1], NorthEast:[n-tcols+2], NorthWest:None,
                                         South:[n+tcols-1], SouthEast:[n+tcols], SouthWest:None,
                                         East:[n+1], West:None})
                        n+=1
                    if c == tcols-2:
                        AdjCells.append({North:[n-tcols+1], NorthEast:None, NorthWest:[n-tcols],
                                         South:[n+tcols-1], SouthEast:None, SouthWest:[n+tcols-2],
                                         East:None, West:[n-1]})
                        n+=1
                    if c>0 and c<tcols-2:    
                        AdjCells.append({North:[n-tcols+1], NorthEast:[n-tcols+2], NorthWest:[n-tcols],
                                         South:[n+tcols-1], SouthEast:[n+tcols], SouthWest:[n+tcols-2],
                                         East:[n+1], West:[n-1]})
                        n+=1        
                
                if r == len(grid)-1:
                    if c == 0:
                        AdjCells.append({North:[n-tcols+1], NorthEast:[n-tcols+2], NorthWest:None,
                                         South:None, SouthEast:None, SouthWest:None,
                                         East:[n+1], West:None})
                        n+=1
                        
                    if c == tcols-2:
                        AdjCells.append({North:[n-tcols+1], NorthEast:None, NorthWest:[n-tcols],
                                         South:None, SouthEast:None, SouthWest:None,
                                         East:None, West:[n-1]})
                        n+=1
                        
                    if c>0 and c<tcols-2:    
                        AdjCells.append({North:[n-tcols+1], NorthEast:[n-tcols+2], NorthWest:[n-tcols],
                                         South:None, SouthEast:None,SouthWest:None,
                                         East:[n+1], West:[n-1]})
                        n+=1
            
            if len(grid)==1:
                if c == 0:
                    AdjCells.append({North:None, NorthEast:None, NorthWest:None,
                                     South:None, SouthEast:None, SouthWest:None,
                                     East:[n+1], West:None})
                    n+=1
                if c == tcols-2:
                    AdjCells.append({North:None, NorthEast:None, NorthWest:None,
                                     South:None, SouthEast:None, SouthWest:None,
                                     East:None,West:[n-1]})
                    n+=1
                if c>0 and c<tcols-2:    
                    AdjCells.append({North:None, NorthEast:None, NorthWest:None,
                                     South:None, SouthEast:None, SouthWest:None,
                                     East:[n+1], West:[n-1]})
                    n+=1
    
    
    return gridcell3, gridcell4, len(grid), tcols-1, AdjCells, CoordCells, cellsize
'''
Returns         arrays of doubles

Inputs:
InFolder        Path to folder (string) 
NCells          int
'''   
# Reads the ASCII files with forest data elevation, saz, slope, and (future) curing degree and returns arrays
# with values
def DataGrids(InFolder, NCells):
    filenames = ["elevation.asc", "saz.asc", "slope.asc", "cur.asc", "cbd.asc", "cbh.asc", "ccf.asc"]
    Elevation =  np.full(NCells, np.nan)
    SAZ = np.full(NCells, np.nan)
    PS = np.full(NCells, np.nan)
    Curing = np.full(NCells, np.nan)
    CBD = np.full(NCells, np.nan)
    CBH = np.full(NCells, np.nan)
    CCF = np.full(NCells, np.nan)
    
    for name in filenames:
        ff = os.path.join(InFolder, name)
        if os.path.isfile(ff) == True:
            aux = 0
            with open(ff, "r") as f:
                filelines = f.readlines()

                line = filelines[4].replace("\n","")
                parts = line.split()

                if parts[0] != "cellsize":
                    print ("line=",line)
                    raise RuntimeError("Expected cellsize on line 5 of "+ ff)
                cellsize = float(parts[1])

                row = 1

                # Read the ASCII file with the grid structure
                for row in range(6, len(filelines)):
                    line = filelines[row]
                    line = line.replace("\n","")
                    line = ' '.join(line.split())
                    line = line.split(" ")
                    #print(line)


                    for c in line: 
                        if name == "elevation.asc":
                            Elevation[aux] = float(c)
                            aux += 1
                        if name == "saz.asc":
                            SAZ[aux] = float(c)
                            aux += 1
                        if name == "slope.asc":
                            PS[aux] = float(c)
                            aux += 1
                        if name == "cbd.asc":
                            CBD[aux] = float(c)
                            aux += 1
                        if name == "cbh.asc":
                            CBH[aux] = float(c)
                            aux += 1
                        if name == "ccf.asc":
                            CCF[aux] = float(c)
                            aux += 1
                        if name == "curing.asc":
                            Curing[aux] = float(c)
                            aux += 1

        # else:
            # print("   No", name, "file, filling with NaN")
            
    return Elevation, SAZ, PS, Curing, CBD, CBH, CCF

'''
Not needed to translate, it is the pandas version of a previous function
Returns         dict {string:int}, dict {int: (double, double, double, double)}

Inputs:
filename        str
'''
# Reads spain_lookup_table.csv and creates dictionaries for the fuel types and cells' colors (Pandas' version) 
# Slower than non pandas version
def Dictionary_PD(filename):
    FbpTable = pd.read_csv(filename, sep=",", index_col=['grid_value'])
    Columns = FbpTable.columns
    FbpTable['extra'] = np.ones(FbpTable.shape[0]) * 255

    FBPDict = dict(FbpTable[Columns[2]].str.replace("-","").str[0:2])
    ColorFBPDict = (FbpTable[[Columns[3],Columns[4],Columns[5],"extra"]] / 255.0).T.to_dict('list')

    return FBPDict, ColorFBPDict

    

'''
Returns         dict{int:int}

Inputs:
filename        str
'''    
# Reads IgnitionPoints.csv file and creates an array with them 
def IgnitionPoints(filename):
    #Ignitions is a dictionary with years = keys and ncell = values
    aux = 1
    file = open(filename, "r") 
    ignitions = {}
    for line in file:
        if aux > 1:
            line = line.replace("\n","")
            line = line.split(",")
            ignitions[int(line[0])] = int(line[1])
        if aux==1:
            aux+=1    
    return ignitions        
    

'''
Returns         dict {string:double}

Inputs
filename        str
nooutput        boolean
'''
# Reads spotting parameters json file and returns dictionary
def ReadSpotting(filename,nooutput):
    # As of Jan 2018 the dictionary should have "SPOT" and "SPTANGLE"
    with open(filename, 'r') as f:
        SpottingParams = json_load_byteified(f)
    ### Thresholds["SPOT"] = 10
    ### Thresholds["SPTANGLE"] = 30
    
    if nooutput == False:
        print("---- Spotting Parameters ----")
        for i in SpottingParams:
            print(i,":",SpottingParams[i])
        print("......................")    
    return SpottingParams


# Ignition cell id = 402647
def id_to_rowcol(ig_id, meta_src, crs):
    row = ig_id // meta_src.meta['width']
    col = ig_id % meta_src.meta['width']

    y, x = meta_src.transform * (col, row)

    ig_gdf = gpd.GeoDataFrame(geometry=[Point(y, x)]).set_crs(crs)

    return ig_gdf

def reset_c2f_results(outputfolder):

    # Reset Cell2Fire input data for re-iteration
    if os.path.exists(outputfolder):
        shutil.rmtree(outputfolder)

# Cell2Fire parameters
def c2f_parameters(
    c2f_path = '/home/minho/research/fire/networks/C2F-W/',
    infolder = '/home/minho/research/fire/networks/C2F-W/data/catalunya_v1/',
    outfolder = '/home/minho/research/fire/networks/C2F-W/results/catalunya_v1_nsim1_iter',
    sim = 'S',
    sim_years = 1,
    nsims = 1, 
    fireperiodlength = 1,
    nweathers = 1000,
    ROS_CV = 0.0,
    ros_threshold = 0.0,
    igradius = 1,
    n_seed = 123,
    nthreads = 8,
    scenario = 3,
    hfactor = 1.0,
    ffactor = 1.0,
    bfactor = 1.0,
    efactor=1.0,
    out_crown=False,
    out_cfb=False,
    out_sfb=False,
    trajectories=False,
    allow_cros = False,
    reset=False,
    verbose=False
    ):

    dataName = os.path.join(infolder, "Data.csv")
    
    # Simulator parameters
    # sim = 'S'                   # Simulator mode (S: Spain)
    # sim_years = 1               # Number of years to simulate
    # nsims = 1                   # Number of simulations
    # fireperiodlength = 0.1      # Fire period length
    # nweathers = 1000            # Number of weather files to select from
    # ROS_CV = 0.0                # Random ROS bias
    # ros_threshold = 0.0         # ROS threshold value
    # igradius = 1                # Ignition radius
    # n_seed = 123                # Random seed
    # nthreads = 8                # Number of threads for parallel processing

    # Compile parameters to run C2F executable
    execArray=[os.path.join(c2f_path,'Cell2FireC/Cell2Fire'),
                '--input-instance-folder', infolder,
                '--output-folder', outfolder,
                '--ignitions',
                '--sim', str(sim),
                '--sim-years', str(sim_years),
                '--nsims', str(nsims),
                '--nthreads', str(nthreads),
                '--grids', 
                '--final-grid',
                '--Fire-Period-Length', str(fireperiodlength),
                '--output-messages',
                '--out-fl',
                '--out-intensity',
                '--out-ros',
                '--weather', 'rows',
                '--nweathers', str(nweathers),
                '--ROS-CV', str(ROS_CV),
                '--seed', str(int(n_seed)),
                '--ROS-Threshold', str(ros_threshold),
                '--scenario', str(scenario),
                '--HFactor', str(hfactor),
                '--BFactor', str(bfactor),
                '--FFactor', str(ffactor),
                '--EFactor', str(efactor),

                ]

    execArray.append('--verbose') if verbose else None
    execArray.append('--cros') if allow_cros else None
    execArray.append('--out_crown') if out_crown else None
    execArray.append('--out_cfb') if out_cfb else None
    execArray.append('--out_sfb') if out_sfb else None
    execArray.append('--trajectories') if trajectories else None


    # Reset Cell2Fire input data for re-iteration
    if reset and os.path.exists(dataName):
        print("Removing data and resetting ...") if verbose else None
        os.remove(dataName)

    if os.path.isfile(dataName) is False:
        print("Generating Data.csv File...",flush=True) if verbose else None
        DataGenerator.GenDataFile(infolder, sim)

    return execArray, dataName   

def record_results(final_gdf, total_merged, n_tactic, n_trial, n_edges, network, tactic_path, output_results):
    
    # Convert network to edges for easy processing
    network_df = nx.to_pandas_edgelist(network)

    # Entire network
    network_avg_ros = np.nanmean(final_gdf['ros'])
    network_total_ros = np.nansum(final_gdf['ros'])
    network_avg_time = np.nanmean(final_gdf['time'])
    network_total_time = np.nansum(final_gdf['time'])

    # Tactic Path (2 superpixels) --> Directed (both ways)
    path_avg_ros = [np.nanmean(network_df[((network_df['source'] == tactic_path[i]) & (network_df['target'] == tactic_path[i+1])) | ((network_df['source'] == tactic_path[i+1]) & (network_df['target'] == tactic_path[i]))]['ros']) for i in range(len(tactic_path)-1)]
    path_total_ros = [np.nansum(network_df[((network_df['source'] == tactic_path[i]) & (network_df['target'] == tactic_path[i+1])) | ((network_df['source'] == tactic_path[i+1]) & (network_df['target'] == tactic_path[i]))]['ros']) for i in range(len(tactic_path)-1)]
    path_avg_time = [np.nanmean(network_df[((network_df['source'] == tactic_path[i]) & (network_df['target'] == tactic_path[i+1])) | ((network_df['source'] == tactic_path[i+1]) & (network_df['target'] == tactic_path[i]))]['time']) for i in range(len(tactic_path)-1)]
    path_total_time = [np.nansum(network_df[((network_df['source'] == tactic_path[i]) & (network_df['target'] == tactic_path[i+1])) | ((network_df['source'] == tactic_path[i+1]) & (network_df['target'] == tactic_path[i]))]['time']) for i in range(len(tactic_path)-1)]
    # path_total_time = [np.nansum(network_df[(network_df['source'] == tactic_path[i]) & (network_df['target'] == tactic_path[i+1])]['time']) for i in range(len(tactic_path)-1)]
    cpath_avg_ros = [np.nanmean(final_gdf[((final_gdf['superpixel_id0'] == tactic_path[i]) & (final_gdf['superpixel_id1'] == tactic_path[i+1])) | ((final_gdf['superpixel_id0'] == tactic_path[i+1]) & (final_gdf['superpixel_id1'] == tactic_path[i]))]['ros']) for i in range(len(tactic_path)-1)]
    cpath_total_ros = [np.nansum(final_gdf[((final_gdf['superpixel_id0'] == tactic_path[i]) & (final_gdf['superpixel_id1'] == tactic_path[i+1])) | ((final_gdf['superpixel_id0'] == tactic_path[i+1]) & (final_gdf['superpixel_id1'] == tactic_path[i]))]['ros']) for i in range(len(tactic_path)-1)]
    cpath_avg_time = [np.nanmean(final_gdf[((final_gdf['superpixel_id0'] == tactic_path[i]) & (final_gdf['superpixel_id1'] == tactic_path[i+1])) | ((final_gdf['superpixel_id0'] == tactic_path[i+1]) & (final_gdf['superpixel_id1'] == tactic_path[i]))]['time']) for i in range(len(tactic_path)-1)]
    cpath_total_time = [np.nansum(final_gdf[((final_gdf['superpixel_id0'] == tactic_path[i]) & (final_gdf['superpixel_id1'] == tactic_path[i+1])) | ((final_gdf['superpixel_id0'] == tactic_path[i+1]) & (final_gdf['superpixel_id1'] == tactic_path[i]))]['time']) for i in range(len(tactic_path)-1)]

    # Full tactic path (All superpixels)
    path_risk = final_gdf.clip(total_merged)
    # print(path_risk['ros'].mean())

    # Record results
    output_results['tactic'].append(n_tactic)
    output_results['proportion'].append(n_edges)
    output_results['source'].append(tactic_path[n_trial])
    output_results['target'].append(tactic_path[n_trial+1])
    output_results['network_avg_ros'].append(network_avg_ros)
    output_results['network_avg_time'].append(network_avg_time)
    output_results['network_total_ros'].append(network_total_ros)
    output_results['network_total_time'].append(network_total_time)
    output_results['path_avg_ros'].append(np.nanmean(path_avg_ros))
    output_results['path_avg_time'].append(np.nanmean(path_avg_time))
    output_results['path_total_ros'].append(np.nansum(path_total_ros))
    output_results['path_total_time'].append(np.nansum(path_total_time))
    output_results['full_path_avg_ros'].append(path_risk['ros'].mean())
    output_results['full_path_avg_time'].append(path_risk['time'].mean())
    output_results['full_path_total_ros'].append(np.nansum(path_risk['ros']))
    output_results['full_path_total_time'].append(np.nansum(path_risk['time']))

    print("--"*30)
    print("Network's average ROS = {:.3f} and average time = {:.3f}".format(network_avg_ros, network_avg_time))    
    print("Only Tactic path's average ROS = {:.3f} and average time = {:.3f}".format(np.nanmean(path_avg_ros), np.nanmean(path_avg_time)))
    print("Comp Tactic path's average ROS = {:.3f} and average time = {:.3f}".format(np.nanmean(cpath_avg_ros), np.nanmean(cpath_avg_time)))
    print("Full tactic path's average ROS = {:.3f} and average time = {:.3f}".format(path_risk['ros'].mean(), path_risk['time'].mean()))

    print('--'*30)
    print("Network's total ROS = {:.3f} and total time = {:.3f}".format(network_total_ros, network_total_time))
    print("Only Tactic path's total ROS = {:.3f} and total time = {:.3f}".format(np.nansum(path_total_ros), np.nansum(path_total_time)))
    print("Comp Tactic path's total ROS = {:.3f} and total time = {:.3f}".format(np.nansum(cpath_total_ros), np.nansum(cpath_total_time)))
    print("Full tactic path's total ROS = {:.3f} and total time = {:.3f}".format(np.nansum(path_risk['ros']), np.nansum(path_risk['time'])))    

    return output_results

