# Geomorphometrics and watershed geometry
# Basin delineation algorithms from PySheds and PyFlwDir

import os, rasterio, math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Geomorphometrics and watershed geometry
from rasterio import features
import geopandas as gpd
from shapely.geometry import Polygon, shape
from rasterio.features import shapes
from shapely.ops import unary_union, transform

import pyflwdir

# Globals
CRS_LATLON = 4326
CRS_UTM = 25831


def quickplot(
    gdfs=[], raster=None, extent=None, hs=None, title="", filename="", aspect=1,
):
    fig = plt.figure(figsize=(8, 15))
    # ax = fig.add_subplot(projection=ccrs.PlateCarree())
    
    ax = fig.add_subplot()    
    
    # plot hillshade background
    if len(hs):
        ax.imshow(
            hs,
            origin="upper",
            extent=extent,
            cmap="Greys",
            alpha=0.3,
            zorder=0,
        )
    # plot geopandas GeoDataFrame
    for gdf, kwargs in gdfs:
        gdf.plot(ax=ax, aspect=aspect, **kwargs)
    if raster is not None:
        data, nodata, kwargs = raster
        ax.imshow(
            np.ma.masked_equal(data, nodata),
            origin="upper",
            extent=extent,
            **kwargs,
        )
    ax.set_aspect("equal")
    ax.set_title(title, fontsize="large")

    if filename:
        plt.savefig(filename, dpi=300)
    return ax

def vectorize(data, nodata, transform, crs=None, name="value"):
    feats_gen = features.shapes(
        data,
        mask=data != nodata,
        transform=transform,
        connectivity=8,
    )
    feats = [
        {"geometry": geom, "properties": {name: val}} for geom, val in list(feats_gen)
    ]

    # parse to geopandas for plotting / writing to file
    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
    gdf[name] = gdf[name].astype(data.dtype)
    return gdf


class Basin:
    def __init__(self, dem_path, buffer=0, plot=False, save=False):
        
        # DEM parameters
        self,
        self.dem_src = rasterio.open(dem_path)
        self.elevation = self.dem_src.read(1).astype(np.float32)
        self.nodata = self.dem_src.nodata
        self.transform = self.dem_src.transform
        self.crs = self.dem_src.crs
        self.extent = np.array(self.dem_src.bounds)[[0, 2, 1, 3]]
        self.latlon = self.dem_src.crs.is_geographic
        self.prof = self.dem_src.profile
        
        # Parameters
        self.buffer = buffer

        # Boolean Flags
        self.plot = plot
        self.save = save

    # Flow direction (D8)
    # If error occurs with heapq, make sure DEM elevation data is set to np.float32
    def _flowdirection(self, invert=False, outlets='min', fdir='d8'):

        elev = 1 - self.elevation if invert else self.elevation
        
        # elev[elev<=0] = np.nan # Filter
        
        # Compute flow direction
        flw = pyflwdir.from_dem(
            data=elev,
            nodata=self.nodata,
            transform=self.transform,
            latlon=self.crs.is_geographic,
            outlets=outlets,
            fdir=fdir
        )
        return flw

    # def flowacc(self)
    
    # Boundary
    def _boundary(self):

        # Mask NaN values from DEM
        mask = ~np.isnan(self.elevation)
        elevtn_masked = self.elevation.copy()
        elevtn_masked[elevtn_masked > 0] = 1    # Set all finite values to 1 (Binary)

        # Step 3: Extract boundaries
        shapes_generator = shapes(elevtn_masked, mask=mask, transform=self.transform)

        # Step 4: Create boundary limit
        geoms = [shape(geom) for geom, value in shapes_generator if value == 1]
        unified_geom = unary_union(geoms)
        
        if isinstance(self.buffer, str) and self.buffer.lower() == 'none':
            self.buffer = None

        try:
            if self.buffer is not None:
                buffer_value = float(self.buffer)  # Convert to float if self.buffer is not None
                if buffer_value > 0:
                    filled_geom = unified_geom.buffer(self.buffer).buffer(self.buffer) # Trick to remove inner holes and retain outer boundary
        except ValueError:
            pass
        else:
            if self.buffer is None:
                filled_geom = unified_geom

        bnd_gdf = gpd.GeoDataFrame(geometry=[filled_geom], crs=self.crs)

        return bnd_gdf

    # Basins
    def _basins(self, flw, pf_depth=4, upstream_area=None, upa_min=1000, n_largest_subbasins=4, img_fname=None, attribute='pfaf2'):
        
        # Erode boundary internally
        # bnd = self._boundary()
        # bnd_gdf_v2 = gpd.GeoDataFrame(geometry=bnd.buffer(-250)).set_crs(self.crs)

        # Delineate basins using the Pfafstetter logic
        pfafbas2, idxs_out = flw.subbasins_pfafstetter(depth=pf_depth, upa_min=upa_min, n_largest_basins=n_largest_subbasins)
        # Vectorize into polygons
        gdf_pfaf2 = vectorize(pfafbas2.astype(np.int32), 0, flw.transform, name=attribute)        # Basin polygons
        gdf_out = gpd.GeoSeries(gpd.points_from_xy(*flw.xy(idxs_out))).set_crs(self.dem_src.crs)     # Basin outlets (points)
        gdf_pfaf2["pfaf"] = gdf_pfaf2["pfaf2"] // 10

        # Clip sub-basins to boundary (mask) extent
        # if self.buffer:
        #     gdf_pfaf2_clipped = gdf_pfaf2.buffer(self.buffer).buffer(-self.buffer).clip(bnd_gdf_v2)
        #     gdf_pfaf2_clipped = gpd.GeoDataFrame(geometry=gdf_pfaf2_clipped).set_crs(CRS_UTM).to_crs(CRS_UTM)
        # else:
        # gdf_pfaf2_clipped = gdf_pfaf2.clip(bnd_gdf_v2)
        # gdf_pfaf2_clipped = gpd.GeoDataFrame(gdf_pfaf2_clipped, geometry='geometry').set_crs(CRS_UTM).to_crs(CRS_UTM)

        gdf_pfaf2_clipped = gdf_pfaf2
        # gdf_pfaf2_clipped = gpd.GeoDataFrame(geometry=gdf_pfaf2_clipped).set_crs(CRS_UTM).to_crs(CRS_UTM)

        gdf_pfaf2_clipped['pfaf'] = gdf_pfaf2['pfaf']
        gdf_pfaf2_clipped['pfaf2'] = gdf_pfaf2['pfaf2']

        # Clip outlet points to extent
        gdf_out_clipped = gpd.GeoDataFrame(geometry=gdf_out.to_crs(CRS_UTM))

        # Plot
        gpd_plot_kwds = dict(
            column="pfaf",
            cmap=matplotlib.cm.Set3_r,
            legend=False,
            categorical=True,
            legend_kwds=dict(title="Pfafstetter code", ncol=3),
            alpha=0.6,
            edgecolor="black",
            linewidth=0.4,
        )

        if self.plot:
            
            # Compute hillshade
            ls = matplotlib.colors.LightSource(azdeg=115, altdeg=45)
            hs = ls.hillshade(np.ma.masked_equal(self.elevation, -9999), vert_exag=1e3)    

            # Plot basins and outlets
            bas = (gdf_pfaf2_clipped, gpd_plot_kwds)
            points = (gdf_out_clipped, dict(color="k", markersize=20))

            title = "Subbasins based on pfafstetter coding (level={})".format(pf_depth)
            ax = quickplot([bas, points], title=title, hs=hs, extent=self.extent, filename=img_fname)
            
            if self.save:
                # Save
                tmp_path = '/data_1/minho/networks/suppression_networks/2024_barcelona'
                gdf_pfaf2_clipped.to_file(os.path.join(tmp_path, 'PFdepth{}_upa{}_basins.shp'.format(pf_depth, upstream_area)))
                gdf_out_clipped.to_file(os.path.join(tmp_path, 'PFdepth{}_upa{}_outlets.shp'.format(pf_depth, upstream_area)))
            
        return gdf_pfaf2_clipped, gdf_out_clipped, flw, pfafbas2, idxs_out


# Flow calculations from Pysheds
def flowdir(dem, dem_src=None, routing='d8', flats=-1, pits=-2, nodata_out=None,
            dirmap=(64, 128, 1, 2, 4, 8, 16, 32), **kwargs):
    """
    Generates a flow direction raster from a DEM grid. Both d8 and d-infinity routing
    are supported.

    Parameters
    ----------
    dem : Raster
            Digital elevation model data.
    flats : int
            Value to indicate flat areas in output array.
    pits : int
            Value to indicate pits in output array.
    nodata_out : int or float
                    Value to indicate nodata in output array.
                    - If d8 routing is used, defaults to 0
                    - If dinf routing is used, defaults to np.nan
    dirmap : list or tuple (length 8)
                List of integer values representing the following
                cardinal and intercardinal directions (in order):
                [N, NE, E, SE, S, SW, W, NW]
    routing : str
                Routing algorithm to use:
                'd8'   : D8 flow directions
                'dinf' : D-infinity flow directions
                'mfd'  : Multiple flow directions

    Additional keyword arguments (**kwargs) are passed to self.view.

    Returns
    -------
    fdir : Raster
            Raster indicating flow directions.
            - If d8 routing is used, dtype is int64. Each cell indicates the flow
                direction defined by dirmap.
            - If dinf routing is used, dtype is float64. Each cell indicates the flow
                angle (from 0 to 2 pi radians).
    """
    default_metadata = {'dirmap' : dirmap, 'flats' : flats, 'pits' : pits}
    input_overrides = {'dtype' : np.float64, 'nodata' : dem_src.nodata}
    kwargs.update(input_overrides)
    
    nodata = dem_src.nodata

    try:
        if np.isnan(nodata):
            nodata_cells = np.isnan(dem).astype(np.bool_)
    except TypeError:
        if nodata is None:
            print("NoData value is None, no need for isnan check.")
            nodata_cells = (dem == nodata).astype(np.bool_)
        else:
            print(f"Handling NoData value as integer or other type: {nodata}")
            # Handle the nodata value as an integer comparison or other type check
            nodata_cells = (dem == nodata).astype(np.bool_)

    if routing.lower() == 'd8':
        if nodata_out is None:
            nodata_out = 0
        
        # Spread direction using Elapsed Time
        fdir = _d8_flowdir(dem, dem_src=dem_src, nodata_cells=nodata_cells,
                            nodata_out=nodata_out, flats=flats,
                            pits=pits, dirmap=dirmap)
    elif routing.lower() == 'dinf':
        if nodata_out is None:
            nodata_out = np.nan
        fdir = _dinf_flowdir(dem=dem, dem_src=dem_src, nodata_cells=nodata_cells,
                                    nodata_out=nodata_out, flats=flats,
                                    pits=pits, dirmap=dirmap)
    else:
        raise ValueError('Routing method must be one of: d8, dinf')
    # fdir.metadata.update(default_metadata)
    return fdir



def _d8_flowdir(dem, dem_src=None, nodata_cells=0, nodata_out=0, flats=-1, pits=-2,
                dirmap=(64, 128, 1, 2, 4, 8, 16, 32)):
    # Make sure nothing flows to the nodata cells
    dem[nodata_cells] = dem.max() + 1
    # Get cell spans and heights
    dx = abs(dem_src.transform.a)
    dy = abs(dem_src.transform.e)
    # Compute D8 flow directions
    fdir = _d8_flowdir_numba(dem, dx, dy, dirmap, nodata_cells,
                                    nodata_out, flat=flats, pit=pits)
    return fdir
    
def _dinf_flowdir(dem, dem_src=None, nodata_cells=0, nodata_out=np.nan, flats=-1, pits=-2,
                    dirmap=(64, 128, 1, 2, 4, 8, 16, 32)):
    # Make sure nothing flows to the nodata cells
    dem[nodata_cells] = dem.max() + 1
    dx = abs(dem_src.transform.a)
    dy = abs(dem_src.transform.e)
    fdir = _dinf_flowdir_numba(dem, dx, dy, nodata_out, flat=flats, pit=pits)

    return fdir

def _d8_flowdir_numba(dem, dx, dy, dirmap, nodata_cells, nodata_out, flat=-1, pit=-2):
    fdir = np.full(dem.shape, nodata_out, dtype=np.int64)
    m, n = dem.shape
    dd = math.sqrt(dx**2 + abs(dy)**2)
    row_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    distances = np.array([dy, dd, dx, dd, dy, dd, dx, dd])
    for i in range(m):
        for j in range(n):
            if not nodata_cells[i, j]:
                elev = dem[i, j]
                max_slope = -np.inf
                for k in range(8):
                    row = i + row_offsets[k]
                    col = j + col_offsets[k]
                    if row < 0 or row >= m or col < 0 or col >= n:
                        # out of bounds, skip
                        continue
                    elif nodata_cells[row, col]:
                        # this neighbor is nodata, skip
                        continue
                    distance = distances[k]
                    slope = (elev - dem[row, col]) / distance
                    if slope > max_slope:
                        fdir[i, j] = dirmap[k]
                        max_slope = slope
                if max_slope == 0:
                    fdir[i, j] = flat
                elif max_slope < 0:
                    fdir[i, j] = pit
    return fdir

# Updated (Working Nov 2024)
def _d8_flowdir_time(arrival_time, dem_src=None, nodata_cells=None, nodata_out=0,
                     flats=-1, pits=-2, dirmap=(64, 128, 1, 2, 4, 8, 16, 32)):
    # Make sure nothing flows to the nodata cells
    arrival_time[nodata_cells] = np.inf  # Use np.inf to represent no flow to nodata cells
    
    # Get cell spans (assuming these are the same as in the original function)
    dx = abs(dem_src.transform.a)
    dy = abs(dem_src.transform.e)

    # Compute D8 flow directions based on fire arrival time
    fdir = _d8_flowdir_numba_fire(arrival_time, dx, dy, dirmap, nodata_cells, nodata_out,
                                    flat=flats, pit=pits)
    return fdir

def _d8_flowdir_numba_fire(arrival_time, dx, dy, dirmap, nodata_cells, nodata_out, flat=-1, pit=-2):
    fdir = np.full(arrival_time.shape, nodata_out, dtype=np.int64)
    m, n = arrival_time.shape
    dd = math.sqrt(dx**2 + dy**2)
    row_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    distances = np.array([dy, dd, dx, dd, dy, dd, dx, dd])
    
    for i in range(m):
        for j in range(n):
            if not nodata_cells[i, j]:
                current_time = arrival_time[i, j]
                min_time = np.inf  # Start with infinity
                for k in range(8):
                    row = i + row_offsets[k]
                    col = j + col_offsets[k]
                    if row < 0 or row >= m or col < 0 or col >= n:
                        continue  # Skip out-of-bounds neighbors
                    elif nodata_cells[row, col]:
                        continue  # Skip nodata neighbors
                    neighbor_time = arrival_time[row, col]
                    distance = distances[k]

                    if neighbor_time < min_time:  # Look for the minimum time
                        fdir[i, j] = dirmap[k]
                        min_time = neighbor_time
                
                if min_time == current_time:  # No lower time found
                    fdir[i, j] = flat  # Flat direction if no time is less
                elif min_time == np.inf:  # All neighbors are nodata
                    fdir[i, j] = pit  # Pit if all neighbors are nodata
                
    return fdir

def _dinf_flowdir_numba(dem, x_dist, y_dist, nodata, flat=-1., pit=-2.):
    m, n = dem.shape
    e1s = np.array([0, 2, 2, 4, 4, 6, 6, 0])
    e2s = np.array([1, 1, 3, 3, 5, 5, 7, 7])
    d1s = np.array([0, 2, 2, 4, 4, 6, 6, 0])
    d2s = np.array([2, 0, 4, 2, 6, 4, 0, 6])
    ac = np.array([0, 1, 1, 2, 2, 3, 3, 4])
    af = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    angle = np.full(dem.shape, nodata, dtype=np.float64)
    diag_dist = math.sqrt(x_dist**2 + y_dist**2)
    cell_dists = np.array([x_dist, diag_dist, y_dist, diag_dist,
                           x_dist, diag_dist, y_dist, diag_dist])
    row_offsets = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    col_offsets = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            e0 = dem[i, j]
            s_max = -np.inf
            k_max = 8
            r_max = 0.
            for k in range(8):
                edge_1 = e1s[k]
                edge_2 = e2s[k]
                row_offset_1 = row_offsets[edge_1]
                row_offset_2 = row_offsets[edge_2]
                col_offset_1 = col_offsets[edge_1]
                col_offset_2 = col_offsets[edge_2]
                e1 = dem[i + row_offset_1, j + col_offset_1]
                e2 = dem[i + row_offset_2, j + col_offset_2]
                distance_1 = d1s[k]
                distance_2 = d2s[k]
                d1 = cell_dists[distance_1]
                d2 = cell_dists[distance_2]
                r, s = _facet_flow(e0, e1, e2, d1, d2)
                if s > s_max:
                    s_max = s
                    k_max = k
                    r_max = r
            if s_max < 0:
                angle[i, j] = pit
            elif s_max == 0:
                angle[i, j] = flat
            else:
                flow_angle = (af[k_max] * r_max) + (ac[k_max] * np.pi / 2)
                flow_angle = flow_angle % (2 * np.pi)
                angle[i, j] = flow_angle
    return angle

def _facet_flow(e0, e1, e2, d1=1., d2=1.):
    s1 = (e0 - e1) / d1
    s2 = (e1 - e2) / d2
    r = math.atan2(s2, s1)
    s = math.hypot(s1, s2)
    diag_angle = math.atan2(d2, d1)
    diag_distance = math.hypot(d1, d2)
    b0 = (r < 0)
    b1 = (r > diag_angle)
    if b0:
        r = 0
        s = s1
    if b1:
        r = diag_angle
        s = (e0 - e2) / diag_distance
    return r, s