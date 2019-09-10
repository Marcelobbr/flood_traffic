import datetime
import pickle
import pytz
import sys
import six
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt

from h3 import h3
from math import ceil
from matplotlib import colors, cm
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scripts.sqlaws as sqlaws
from scripts.mod_simplify import simplify_graph

from pathlib import Path 
current_path = Path().resolve()
RAW_PATH = current_path.parent / 'data' / 'raw'
OUTPUT_PATH = current_path.parent / 'data' / 'output' 

## Get Data
def download_potholes_data(template_fp, city, table, start_date, end_date, s3output_bucket='athena-fgv', s3output_prefix='pot-holes/',
                           workgroup='primary', database='cities'):

    try:
        sd, ed = start_date, end_date
        csv_localpath = RAW_PATH/city/f'Athena-pot-holes-{city}-from-{sd.year}-{sd.month}-{sd.day}-to-{ed.year}-{ed.month}-{ed.day}.csv'
        pot_holes = pd.read_csv(csv_localpath)
        cols_to_drop = [col for col in pot_holes.columns if 'Unnamed' in col]
        pot_holes = pot_holes.drop(columns=cols_to_drop)
    except:
        pot_holes = sqlaws.query_aws(template_fp=template_fp, s3output_bucket=s3output_bucket, s3output_prefix=s3output_prefix, 
                              workgroup=workgroup, database=database, city=city, table=table, start_date=start_date, end_date=end_date)
        
        cols_to_drop = [col for col in pot_holes.columns if 'Unnamed' in col]
        pot_holes = pot_holes.drop(columns=cols_to_drop)
        pot_holes.to_csv(csv_localpath)
    
    return pot_holes

def download_osm_graph(city, osm_place=None, shapefile=None, north=None, south=None, east=None, west=None,
                        simplify=True, cached=True, cache_result=True, which_result=1):
    try:
        if not cached:
            raise
        if simplify:
            G = ox.save_load.load_graphml(filename=(city+'.graphml'), folder=str(RAW_PATH/city))
        else:
            G = ox.save_load.load_graphml(filename=(city+'_ns.graphml'), folder=str(RAW_PATH/city))
    except:
        if osm_place:
            G = ox.graph_from_place(osm_place, network_type='drive', name=city, 
                                    retain_all=True, simplify=simplify, which_result=which_result)
        elif shapefile:
            polygon = gpd.read_file(shapefile)['geometry'].values[0]
            G = ox.graph_from_polygon(polygon, network_type='drive', name=city, retain_all=True, simplify=simplify)
        elif north and south and east and west:
            buffer_dist = 500
            polygon = Polygon([(west, north), (west, south), (east, south), (east, north)])
            polygon_utm, crs_utm = ox.project_geometry(geometry=polygon)
            polygon_proj_buff = polygon_utm.buffer(buffer_dist)
            polygon_buff, _ = ox.project_geometry(geometry=polygon_proj_buff, crs=crs_utm, to_latlong=True)
            west_buffered, south_buffered, east_buffered, north_buffered = polygon_buff.bounds
            G = ox.graph_from_bbox(north=north_buffered, south=south_buffered, east=east_buffered, west=west_buffered, 
                                    network_type='drive', name=city, retain_all=True, simplify=simplify)
        else:
            raise Exception("No cached Graph. Must pass osm_place or shapefile to retrive city's Graph")
        if cache_result:
            if simplify:
                ox.save_load.save_graphml(G, filename=(city+'.graphml'), folder=str(RAW_PATH/city))
            else:
                ox.save_load.save_graphml(G, filename=(city+'_ns.graphml'), folder=str(RAW_PATH/city))
    return G


## Treat data

def treat_alerts_points(alerts):
    
    return gpd.GeoDataFrame(alerts, geometry=gpd.points_from_xy(alerts['longitude'], alerts['latitude']), crs="+init=epsg:4326")
    

## Match roads to points

def get_extended_edges(city, edges=None, batch_size=10000, hex_res=8, cached=True, cache_result=True):

    def apply_redistribute_vertices(proj_edges, dist=10):
        edges = proj_edges.copy()
        edges = edges.loc[:,['u','v','key','geometry']]
        edges['points'] = edges.apply(lambda x: ox.redistribute_vertices(x.geometry, dist), axis=1)
        edges = edges['points'].apply([pd.Series]).stack().reset_index(level=1, drop=True).join(edges).reset_index()\
                .drop(columns=['points', 'geometry']).rename(columns={'Series': 'points'})
        return edges

    try:
        if not cached:
            raise
        with open(str(RAW_PATH/city/f'extended_edges_hex_res{hex_res}.p'), 'rb') as f:
            extended = pickle.load(f)
    except:
        assert edges is not None, "edges DataFrame must be provided"  
        batches = ceil(edges.shape[0] / batch_size)

        extended = pd.DataFrame()
        for i in range(batches):
            extended = extended.append(
                apply_redistribute_vertices(edges.iloc[i*batch_size:(i+1)*batch_size, :]), 
                ignore_index=True)

        extended = extended.rename(columns={'points':'proj_points'})
        extended = extended.set_geometry('proj_points')
        extended.crs = edges.crs
        
        extended['points'] = extended['proj_points'].to_crs('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
        extended['lon'] = extended.apply(lambda row: row["points"].x, axis=1)
        extended['lat'] = extended.apply(lambda row: row["points"].y, axis=1)
        extended["hex_id"] = extended.apply(lambda row: h3.geo_to_h3(row["lat"], row["lon"], hex_res), axis = 1)
        extended.hex_res = hex_res

        if cache_result:
            with open(str(RAW_PATH/city/f'extended_edges_hex_res{hex_res}.p'), 'wb') as f:
                pickle.dump(extended, f)

            extended.drop(columns='points').to_csv(OUTPUT_PATH/city/"extended_edges.csv")
        
    return extended

def get_hex_KDTrees(city, res=8, extended=None, cached=True, cache_result=True):

    def get_edges_idx_in_hex(hex_id, extended):
        return extended.loc[extended.hex_id == hex_id, 'index'].unique()

    def get_extended_idx_in_ring(hex_ring, extended):
        return extended.loc[extended.hex_id.isin(hex_ring)].index

    def construct_KDTree(hex_ring, extended, extended_idx):
        points = extended.loc[extended_idx, 'proj_points']
        nbdata = np.array(list(zip(points.apply(lambda point: point.x),
                                   points.apply(lambda point: point.y))))
        return cKDTree(data=nbdata, compact_nodes=True, balanced_tree=True)

    try:
        if not cached:
            raise
        with open(str(RAW_PATH/city/f'h3_hex{res}_KDTree.p'), 'rb') as f:
            h3_hex = pickle.load(f)
    except:
        assert extended is not None, "extended DataFrame must be provided"
        assert extended.hex_res == res, "extended hex_res differ from res"
        h3_hex = pd.DataFrame(extended["hex_id"].unique(), columns=['hex_id']).set_index('hex_id', drop=False)
        h3_hex["hex_ring1"] = h3_hex["hex_id"].apply(h3.k_ring, 1, args=(1,))
        h3_hex['edges_idx_in_hex'] = h3_hex['hex_id'].apply(get_edges_idx_in_hex,args=(extended,))
        h3_hex['extended_idx_in_ring'] = h3_hex['hex_ring1'].apply(get_extended_idx_in_ring,args=(extended,))
        h3_hex['KDTree'] = h3_hex.apply(lambda row: construct_KDTree(row['hex_ring1'], extended, row['extended_idx_in_ring']),1)
        h3_hex['geometry'] = h3_hex.hex_id.apply(lambda hid: Polygon(h3.h3_to_geo_boundary(hid, geo_json=True)))
        h3_hex = gpd.GeoDataFrame(h3_hex)
        h3_hex.crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        
        if cache_result:
            with open(str(RAW_PATH/city/f'h3_hex{res}_KDTree.p'), 'wb') as f:
                pickle.dump(h3_hex, f)

    return h3_hex


def match_edges_to_points_by_hex(city, alerts, G_proj, hex_res=8):
    
    edges = ox.graph_to_gdfs(G_proj, nodes=False, fill_edge_geometry=True)

    def query_KDTree(row, alerts_proj, alerts_hexs, extended):
        hex_id = row['hex_id']
        # Some hexagons in the city may not have reported pot-holes        
        if hex_id not in alerts_hexs:
            return np.array([], dtype=np.int64).reshape(0,5)
        alerts_byhex = alerts_proj.loc[alerts_proj.hex_id == hex_id]
        X = alerts_byhex['geometry'].apply(lambda row: row.x).values 
        Y = alerts_byhex['geometry'].apply(lambda row: row.y).values

        points = np.array([X, Y]).T
        _, idx = row['KDTree'].query(points, k=1)  # Returns ids of closest point
        eidx = extended.loc[row['extended_idx_in_ring']].reset_index(drop=True).loc[idx, 'index']
        ne = edges.loc[eidx, ['u', 'v','key']]
        
        # np.zeros(ne.shape[0], dtype=int)
        return np.c_[ne , alerts_byhex[['uuid', 'interactions']].values]    
    
    extended = get_extended_edges(city, edges, hex_res=hex_res)

    h3_hex = get_hex_KDTrees(city, extended, res=hex_res)
    
    alerts['hex_id'] = alerts.apply(lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], hex_res), axis = 1)
    alerts_proj = alerts.to_crs(edges.crs)
    alerts_hexs = set(alerts_proj['hex_id'].values)
    
    edges_potholes = np.vstack(tuple(h3_hex.apply(query_KDTree, 1, args=(pot_holes_proj, pot_holes_hexs, extended)).values))
    edges_potholes = pd.DataFrame(edges_potholes, columns=['u','v','key','alerts_count','interactions'])

    return edges_potholes.groupby(['u','v','key']).agg({'alerts_count': 'count', 'interactions': 'sum'}).join(
        extended.groupby(['u','v','key'])['hex_id'].apply(set), how='outer').rename(columns={'hex_id':'hex_set'}).to_dict('index')



def match_roads_to_points(city, alerts, G_proj, by=None, nearest_edges=False):
    '''
    Use projected non-simplified Graph for better accuracy in identifying pothole location
    '''
    if by and by[:3] == 'hex':
        if by[3:]:
            return match_edges_to_points_by_hex(city, alerts, G_proj, hex_res=int(by[3:]))
        return match_edges_to_points_by_hex(city, alerts, G_proj)

    alerts_proj = alerts.to_crs(G_proj.graph['crs'])
    
    X = alerts_proj['geometry'].apply(lambda row: row.x).values 
    Y = alerts_proj['geometry'].apply(lambda row: row.y).values 

    edges_potholes = ox.utils.get_nearest_edges(G_proj, X, Y, method='kdtree', dist=10)
    
    edges_potholes = np.c_[edges_potholes, np.zeros(edges_potholes.shape[0], dtype=int), 
                              alerts_proj[['uuid', 'interactions']].values]

    if nearest_edges:
        return pd.DataFrame(edges_potholes, columns=['u','v','k','uuid','interactions'])

    edges_potholes = pd.DataFrame(edges_potholes, columns=['u','v','k','alerts_count','interactions'])

    edges_potholes = edges_potholes.groupby(['u','v','k']).agg({'alerts_count': 'count', 'interactions': 'sum'}).to_dict('index')
    
    # Caching
    #alerts.to_csv(OUTPUT_PATH / city / 'alerts_to_segments.csv')
    #edges.to_csv(OUTPUT_PATH /  city /  'potholes_full.csv')
    
    return edges_potholes

def project_alert_on_nearest_edge(city, alerts, G_proj):

    edges = ox.graph_to_gdfs(G_proj, nodes=False, fill_edge_geometry=True).set_index(['u','v','key'])
    alerts = alerts.to_crs(G_proj.graph['crs'])

    nearest_edges = match_roads_to_points(city, alerts, G_proj, nearest_edges=True)

    alerts = alerts.set_index('uuid')

    nearest_edges['line'] = nearest_edges.apply(lambda row: edges.loc[(row['u'], row['v'],row['k']), 'geometry'], 1)
    nearest_edges['point'] = nearest_edges.uuid.apply(lambda uuid: alerts.loc[uuid, 'geometry'], 1)
    
    return nearest_edges.apply(lambda row: row['line'].interpolate(row['line'].project(row['point'])), 1).values


## Plotting

def plot_graph_potholes(G, fig_width=10, fig_height=10):
    interactions = [d['interactions'] for _,_,d in G.edges(data=True)]
    norm = colors.Normalize(vmin=0, vmax=max(interactions))
    scalarMap  = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('autumn_r'))

    edges_color = list(map(colors.to_rgba, [scalarMap.to_rgba(d['interactions']) if d['interactions'] > 0 else '#bfbfbf' for _,_,d in G.edges(data=True)]))
    edges_width = [2.5 if d['alerts_count'] > 0 else 0.2 for _,_,d in G.edges(data=True)]
    
    fig, ax = ox.plot_graph(G, node_size=0, fig_width=fig_width, fig_height=fig_height, edge_color=edges_color,
                            edge_linewidth=edges_width, show=False, axis_off=False)
    ax.set_facecolor('k')
    ax.margins(0)
    ax.tick_params(which='both', left=False, bottom=False,  labelleft=False, labelbottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(scalarMap, cax=cax)
    cb.set_label('Quantidade Total de Interações')
    #ax1.set_xlabel(xlabel)
    return fig