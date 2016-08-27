# -*- coding: utf-8 -*-

"""
 helpers.py
 Richard Wen (rwenite@gmail.com)
 
===============================================================
 
 A module of useful helper functions.
 
===============================================================
"""


"""
===============================================================
 Modules
===============================================================
"""


from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


import geopandas as gpd
import numpy as np
import pandas as pd
import rtree


"""
===============================================================
 Variables
===============================================================
"""


_sep = '_'


"""
===============================================================
 Functions
===============================================================
"""


def _get_row_vertices(row, geometry_index=0, geom_type_index=1):
    """
     _get_row_vertices: pd.Series int int -> int OR None
     
    ---------------------------------------------------------------
     
     Takes a row from a pandas series with geopandas geometry
     and geom_type columns, and returns the number of vertices
     in the feature row. This will return vertices for geometries
     of type LineString, Polygon, and Point.
     
     If there is no geometry for a feature to count vertices on, then
     None will be given.
     
     Notes
     -----
     * Created for purpose of pandas dataframe apply function
       with stored geopandas geometry and geom_type cols
     
     Required Parameters
     -------------------
     * row: pd.Series
             A row of data in pandas series format containing a
             geopandas 'geometry' col with shapely geometry data
             and a 'geom_type' col specifying the type of geometry
             it is.
             
             The default format assumes that the 1st col is the
             geometry col and the 2nd col is the geom_type col.
             
             Default Format:
                  geometry  |  geom_type  | other..
                ------------|-------------|---------
                shapely obj |    str      | any
                
     Optional Parameters
     -------------------
     * geometry_index: int
             The index number of the geometry col
     * geom_type_index: int
             The index number of the geom_type col
             
     Returns
     -------
     * : int
             The number of vertices for the row of geometry
             
     Examples
     --------
     * Read a spatial data file
         import geopandas as gpd
         import pandas as pd
         geodata = gpd.read_file('path/to/file')
         geodata_concat = pd.concat([geodata.geometry, geodata.geom_type], axis = 1)
     * Get the vertice count for a row in geodata
         vertices = _get_row_vertices(geodata_concat.ix[0])

    ---------------------------------------------------------------
    """
    geometry = row.iloc[geometry_index]
    geom_type = row.iloc[geom_type_index].lower()  
    if geom_type == 'point':
        return 1
    elif geom_type == 'polygon':
        return len(geometry.exterior.coords)
    elif geom_type == 'linestring':
        return len(geometry.coords)
    else:
        return None
        
        
def concat_gdf(gdfs, crs=None, epsg='4326', reset_index=True, *args, **kwargs):
    """
     concat_gdf: ((listof gpd.GeoDataFrame) OR (listof str))
                 dict
                 str
                 str OR None
                 bool
                 -> gpd.GeoDataFrame
     
    ---------------------------------------------------------------
     
     Combines a list of geopandas geodataframe objects into a single
     merged geodataframe. The coordinate reference system (crs) of the
     list of geodataframes or list of geodata paths will all be projected
     into [epsg] or [crs] respectively.
     
     Required Parameters
     -------------------
     * gdfs: (listof gpd.GeoDataFrame) OR (listof str)
             A list of geopandas geodataframes or paths to readable geopandas
             geodataframes to be combined; the crs attributes of these must be
             the same otherwise projecting the merged dataframe is not
             consistent.

     Optional Parameters
     -------------------
     * crs: dict
             The coordinate reference system to be set in dictionary form.
             http://geopandas.org/user.html#GeoSeries.to_crs
     * epsg: str
             The coordinate reference system number in epsg to project the data to.
             http://geopandas.org/user.html#GeoSeries.to_crs
     * reset_index: bool
             Whether to reset the index of the combined geodataframe or not.
     * *args: args
             Arguments to be passed to pd.concat function after argument 'objs'
             See http://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html
     * **kwargs: kwargs
             Keyword arguments to be passed to pd.concat function after after argument 'objs'
             See http://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html
    
     Returns
     -------
     * gdf: gpd.GeoDataFrame
             The combined geopandas geodataframe from the [gdfs]
     
     Examples
     --------
     * Combine all geodataframes in gdf list by row
         import geopandas as gpd
         gdfs = ['path/to/inputFile1.geojson', 'path/to/inputFile2.shp']
         gdf = concat(gdfs)
     
    ---------------------------------------------------------------
    """
    gdf_list = [gpd.read_file(g).to_crs(crs=crs, epsg=epsg) for g in gdfs] if all(isinstance(g, str) for g in gdfs) else [g.to_crs(crs=crs, epsg=epsg) for g in gdfs]
    df = pd.concat(gdf_list, *args, **kwargs)
    if reset_index:
        df = df.reset_index(drop=True)
    gdf = gpd.GeoDataFrame(df)
    gdf.geometry = gpd.GeoSeries(gdf.geometry)
    gdf.crs = gdf_list[0].crs
    return gdf
    
    
def get_series(df, series_cols, by_row=False, check=True):
    """
     get_series: pd.DataFrame
                     (listof str)
                     bool
                     bool
                     -> pd.DataFrame
     
    ---------------------------------------------------------------
     
     Extract selected series or attributes (returning series)
     from the [df] geodataframe.
     
     Required Parameters
     -------------------
     * df: pd.DataFrame
             A pandas dataframe object or extended class object.
     * series_cols: (listof str)
             A list of column names for the series to be extracted.
             These series can exist in the dataframe itself or as an attribute
             of the dataframe object. The series columns in the dataframe will
             be prioritized over attributes.

     Optional Parameters
     -------------------
     * by_row: bool
             Whether to combine the extracted columns by row.
     * check: bool
             Whether to raise an error if any of the [series_cols]
             are not either a series or an attribute
    
     Returns
     -------
     * extract_data: pd.DataFrame
             The extracted series data as a pandas dataframe
             containing only the attributes and cols selected
     
     Examples
     --------
     * Read data from a file
         import pandas as pd
         data = pd.read_csv('path/to/inputFile.csv')
     * Extract columns b and c only
         extract_series(data, series_cols = ['b', 'c'])
          a | b | c             b  |  c
        ----|---|---     ->    ----|----
          1 | 0 | 1             0  |  1
          2 | 1 | 0             1  |  0
          3 | 1 | 1             1  |  1
     
    ---------------------------------------------------------------
    """
    by_row = int(not by_row)  # for pandas axis (1 for cols, 0 for rows)
    
    # (Extract_Data) Loop through each series as a col
    series = []
    for col in series_cols:
        if col in df.columns and type(col) == pd.Series:
            series.append(df[col])
        elif hasattr(df, col):
            series.append(getattr(df, col))
        elif check:
            raise Exception('Column or attribute (' + col + ') does not exist')
    
    # (Return) Returns the extracted series and attr
    extract_data = pd.concat(series, axis=by_row)
    extract_data.columns = series_cols
    return extract_data
    
    
def get_vtx(geo):
    """
     get_vtx: gpd.GeoDataFrame OR gpd.GeoSeries
              -> pd.Series
     
    ---------------------------------------------------------------
     
     Returns the number of vertices as a pandas series
     given [geo.geometry] data and [geo.geom_type].
     
     Required Parameters
     -------------------
     * geo: gpd.GeoDataFrame OR gpd.GeoSeries
             A geopandas dataframe/geoseries object or extended class object.
    
     Returns
     -------
     * vtx: pd.Series
             A pandas series containing the sum of vertices for
             [geo] given data on [geo.geometry] and [geo.geom_type]
    
     Examples
     --------
     * Read a spatial data file
         import geopandas as gpd
         geodata = gpd.read_file('path/to/file')
     * Get the vertice count for each feature in geodata
         vertex_series = get_vertices(geodata)
             
    ---------------------------------------------------------------
    """
    geom = pd.concat([geo.geometry, geo.geom_type], axis=1)
    vtx = geom.apply(_get_row_vertices, axis=1)
    vtx.name = 'vertices'
    return vtx
    
    
def nb_dist(origin, near=None, fill=-1.0, eq=False, name='nb_dist'):
    """
     nb_dist: gpd.GeoSeries OR gpd.GeoDataFrame
              gpd.GeoSeries OR gpd.GeoDataFrame OR None
              int
              bool
              str
              -> pd.Series
     
    ---------------------------------------------------------------
     
     Return a dataframe of the nearest features from [near]
     for each feature that is within the bounds of [origin].
     If [near] is not provided, the [origin] geometries will take its place.
     
     If there are no features within the bounds of a feature in
     [origin], the [fill] value will be used.
     
     Required Parameters
     -------------------
     * origin: gpd.GeoSeries OR gpd.GeoDataFrame
             The geoseries or geodataframe where each feature
             is checked for the [near] feature distance.

     Optional Parameters
     -------------------
     * near: gpd.GeoSeries OR gpd.GeoDataFrame OR None
             The geoseries or geodataframe of nearest feature distances
             to calculate for each feature in [origin]. If
             None, then the [origin] geometries will be used instead.
     * fill: int
             Number to represent that there weren't any [near]
             features within the bounds of [origin]
     * eq: bool
             Whether to include matching row indices of [origin]
             and [near] as a possible nearest feature for distance
             calculation.
     * name: str
             A string to name the [near_dists] series
             
     Returns
     -------
     * near_dists: pd.Series
             The nearest distance of [near] feature for each [origin] feature
             if the [near] feature is within the bounds of [origin]
             Format:
                row ids of [origin]..  |   dist to [near]..
                
     Examples
     --------
     * Obtain the spatial relations of g1 for matching g2
         from geopandas import GeoSeries
         from shapely.geometry import Polygon
         p1 = Polygon([(0, 0), (1, 0), (1, 1)])
         p2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
         p3 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
         g1 = GeoSeries([p1, p2, p3])
         g2 = GeoSeries([p2, p3])
         sr = nb_dist(g1, g2)
     
    ---------------------------------------------------------------
    """
    bar = name + ': {l_bar}{bar}{r_bar}'
    near = origin.geometry if near is None else near.geometry
    
    # (Rtree_Index) Create a spatial index using rtree
    sidx = rtree.index.Index()
    for i, g in tqdm(zip(near.index, near), unit='/sidx', total=len(near), bar_format=bar):
        sidx.insert(i, g.bounds)
    
    # (Calc_SR) Calculate spatial relations matrix
    near_dists = pd.Series(fill, index=origin.index, name=name)
    for i, g in tqdm(zip(origin.index, origin.geometry), unit='/geom', total=len(origin.geometry), bar_format=bar):
        idx = list(sidx.nearest(g.bounds, 2))
        if not eq and i in idx:
            idx.remove(i)
        if len(idx) > 0:
            near_dists.loc[i] = near[idx].distance(g).iloc[0]
        else:
            near_dists.loc[i] = near.distance(g).iloc[0]
    return near_dists
    
    
def ocorr_df(df, neg=-0.8, pos=0.8):
    """
     ocorr_df: pd.DataFrame num num -> pd.DataFrame
     
    ---------------------------------------------------------------
     
     Creates an ordered correlation dataframe for the removal of highly
     correlated variables in [df]. The ordered dictionary is created
     by going through each [df] var in order, and removing its highly correlated
     pair until all correlated pairs of variables are removed from [df].
     
     It is suggested to reorder the cols of [df] based on importance
     to obtain more desirable variables to remove multicollinearity in the [df].
          
     Required Parameters
     -------------------
     * df: pd.DataFrame
             The dataframe to generate the ordered correlation dictionary
             with.

     Optional Parameters
     -------------------
     * neg: num
             The correlation range to determine highly negatively
             correlated variables. Any correlation value less than or equal
             to this value will be considered negatively highly correlated.
     * pos: num
             The correlation range to determine highly positively
             correlated variables. Any correlation value greater than or equal
             to this value will be considered positively highly correlated.
    
     Returns
     -------
     * ocorr: pd.DataFrame
             The ordered correlation removal dataframe in which there are cols of
             the variables to keep and variables to remove based on
             high correlations to its correspoding keep variable.
             
             Format:
                       keep      |                remove
                -----------------|---------------------------------------
                  vars to keep.. |    rmv vars for high corr to keep..
     
    ---------------------------------------------------------------
    """
    corr = df.corr()
    np.fill_diagonal(corr.values, 0)
    hcorr = []
    keep = []
    remove = []
    for v in corr.columns:
        if v not in hcorr:
            hcv = corr[v][(corr[v] >= pos) | (corr[v] <= neg)].index.tolist()
            hcorr += hcv
            keep.append(v)
            remove.append(hcv)
    return pd.DataFrame({'keep': keep, 'remove': remove})
    
    
def rf_prox(forest, X):
    """
     rf_prox: obj array -> array
     
    ---------------------------------------------------------------
     
     Returns the proximity measures for the random forest on a set
     of training samples after the [forest] is fit to the data.
     
     Credits
     -------
     * Gilles Louppe [1]: author of function
     
     References
     ----------
     * [1] Gilles Louppe @ University of Leige:
       https://github.com/glouppe/phd-thesis/blob/master/scripts/ch4_proximity.py
     
     Required Parameters
     -------------------
     * forest: obj
             A RandomForestClassifier or RandomForestRegressor object
             from sklearn that has been fitted to some training data
     * X: array
             The data to calculate proximity measures on. It must be
             structured the same as the training data used to fit
             the [forest] classifier or regressor.
    
     Returns
     -------
     * prox: array
             An array of proximity values presented as a matrix in which
             the proximity of each sample is matched to all samples.
             
    ---------------------------------------------------------------
    """
    prox = pdist(forest.apply(X), lambda u, v: (u == v).sum()) / forest.n_estimators
    prox = squareform(prox)
    return prox
    
    