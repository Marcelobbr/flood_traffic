import bz2
import datetime as dt
import io
import logging as lg
import math
import os
import networkx as nx
import numpy as np
import pandas as pd
import requests
import sys
import time
import unicodedata
import warnings
import xml.sax
from collections import Counter
from itertools import chain
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon

import osmnx as ox

from osmnx import settings

# scipy and sklearn are optional dependencies for faster nearest node search
try:
    from scipy.spatial import cKDTree
except ImportError as e:
    cKDTree = None
try:
    from sklearn.neighbors import BallTree
except ImportError as e:
    BallTree = None


def get_nearest_edges(G, X, Y, method=None, dist=0.0001):
    """
    Return the graph edges nearest to a list of points. Pass in points
    as separate vectors of X and Y coordinates. The 'kdtree' method
    is by far the fastest with large data sets, but only finds approximate
    nearest edges if working in unprojected coordinates like lat-lng (it
    precisely finds the nearest edge if working in projected coordinates).
    The 'balltree' method is second fastest with large data sets, but it
    is precise if working in unprojected coordinates like lat-lng.
    Parameters
    ----------
    G : networkx multidigraph
    X : list-like
        The vector of longitudes or x's for which we will find the nearest
        edge in the graph. For projected graphs, use the projected coordinates,
        usually in meters.
    Y : list-like
        The vector of latitudes or y's for which we will find the nearest
        edge in the graph. For projected graphs, use the projected coordinates,
        usually in meters.
    method : str {None, 'kdtree', 'balltree'}
        Which method to use for finding nearest edge to each point.
        If None, we manually find each edge one at a time using
        osmnx.utils.get_nearest_edge. If 'kdtree' we use
        scipy.spatial.cKDTree for very fast euclidean search. Recommended for
        projected graphs. If 'balltree', we use sklearn.neighbors.BallTree for
        fast haversine search. Recommended for unprojected graphs.
    dist : float
        spacing length along edges. Units are the same as the geom; Degrees for
        unprojected geometries and meters for projected geometries. The smaller
        the value, the more points are created.
    Returns
    -------
    ne : ndarray
        array of nearest edges represented by their startpoint and endpoint ids,
        u and v, the OSM ids of the nodes.
    Info
    ----
    The method creates equally distanced points along the edges of the network.
    Then, these points are used in a kdTree or BallTree search to identify which
    is nearest.Note that this method will not give the exact perpendicular point
    along the edge, but the smaller the *dist* parameter, the closer the solution
    will be.
    Code is adapted from an answer by JHuw from this original question:
    https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point
    -in-other-dataframe
    """
    start_time = time.time()

    if method is None:
        # calculate nearest edge one at a time for each point
        ne = [get_nearest_edge(G, (x, y)) for x, y in zip(X, Y)]
        ne = [(u, v) for _, u, v in ne]

    elif method == 'kdtree':

        # check if we were able to import scipy.spatial.cKDTree successfully
        if not cKDTree:
            raise ImportError('The scipy package must be installed to use this optional feature.')

        # transform graph into DataFrame
        edges = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

        # transform edges into evenly spaced points
        edges['points'] = edges.apply(lambda x: redistribute_vertices(x.geometry, dist), axis=1)

        # develop edges data for each created points
        extended = edges['points'].apply([pd.Series]).stack().reset_index(level=1, drop=True).join(edges).reset_index()

        # Prepare btree arrays
        nbdata = np.array(list(zip(extended['Series'].apply(lambda x: x.x),
                                   extended['Series'].apply(lambda x: x.y))))

        # build a k-d tree for euclidean nearest node search
        btree = cKDTree(data=nbdata, compact_nodes=True, balanced_tree=True)

        # query the tree for nearest node to each point
        points = np.array([X, Y]).T
        dist, idx = btree.query(points, k=1)  # Returns ids of closest point
        eidx = extended.loc[idx, 'index']
        ne = edges.loc[eidx, ['u', 'v', 'key']]

    elif method == 'balltree':

        # check if we were able to import sklearn.neighbors.BallTree successfully
        if not BallTree:
            raise ImportError('The scikit-learn package must be installed to use this optional feature.')

        # transform graph into DataFrame
        edges = graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

        # transform edges into evenly spaced points
        edges['points'] = edges.apply(lambda x: redistribute_vertices(x.geometry, dist), axis=1)

        # develop edges data for each created points
        extended = edges['points'].apply([pd.Series]).stack().reset_index(level=1, drop=True).join(edges).reset_index()

        # haversine requires data in form of [lat, lng] and inputs/outputs in units of radians
        nodes = pd.DataFrame({'x': extended['Series'].apply(lambda x: x.x),
                              'y': extended['Series'].apply(lambda x: x.y)})
        nodes_rad = np.deg2rad(nodes[['y', 'x']].values.astype(np.float))
        points = np.array([Y, X]).T
        points_rad = np.deg2rad(points)

        # build a ball tree for haversine nearest node search
        tree = BallTree(nodes_rad, metric='haversine')

        # query the tree for nearest node to each point
        idx = tree.query(points_rad, k=1, return_distance=False)
        eidx = extended.loc[idx[:, 0], 'index']
        ne = edges.loc[eidx, ['u', 'v']]
    
    else:
        raise ValueError('You must pass a valid method name, or None.')

    #log('Found nearest edges to {:,} points in {:,.2f} seconds'.format(len(X), time.time() - start_time))

    return np.array(ne), idx, extended
        
        
        
def redistribute_vertices(geom, dist):
    """
    Redistribute the vertices on a projected LineString or MultiLineString. The distance
    argument is only approximate since the total distance of the linestring may not be
    a multiple of the preferred distance. This function works on only [Multi]LineString
    geometry types.
    This code is adapted from an answer by Mike T from this original question:
    https://stackoverflow.com/questions/34906124/interpolating-every-x-distance-along-multiline-in-shapely
    Parameters
    ----------
    geom : LineString or MultiLineString
        a Shapely geometry
    dist : float
        spacing length along edges. Units are the same as the geom; Degrees for unprojected geometries and meters
        for projected geometries. The smaller the value, the more points are created.
    Returns
    -------
        list of Point geometries : list
    """
    if geom.geom_type == 'LineString':
        num_vert = int(round(geom.length / dist))
        if num_vert == 0:
            num_vert = 1
        return [geom.interpolate(float(n) / num_vert, normalized=True)
                for n in range(num_vert + 1)]
    elif geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, dist)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry {}'.format(geom.geom_type))