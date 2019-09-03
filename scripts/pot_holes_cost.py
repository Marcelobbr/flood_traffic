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

abs_path = '/home/master/cts/cities/waze-tools'
sys.path.append(abs_path)

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

def download_interactions_data(city, cities):
    """Deprected"""
    
    query = """
    select DISTINCT uuid, MAX(thumbs_up)
    from waze.alerts
    where subtype = 'HAZARD_ON_ROAD_POT_HOLE'
    and city = '{city}'
    group by uuid
    """.format(city=city)
    cities[city]['potholes_interactions'] = pd.read_sql_query(query, con=con)
    cities[city]['potholes_interactions'].columns = ['uuid', 'interactions']
    return cities

def get_cities_metadata():
    """Deprected"""
    return pd.read_sql_table('cities_stats', schema='waze', con=con).set_index('city_as_waze').to_dict('index')

def get_timerange(city):

    """Deprected"""
    
    query = """
    select MIN(pub_utc_date) as "initial date", MAX(pub_utc_date) as "final date"
    from waze.alerts
    where subtype = 'HAZARD_ON_ROAD_POT_HOLE'
    and city = '{city}'
    """.format(city=city)
    return pd.read_sql_query(query, con=con)

## Treat data

def treat_potholes_points(pot_holes):
    
    return gpd.GeoDataFrame(pot_holes, geometry=gpd.points_from_xy(pot_holes['longitude'], pot_holes['latitude']), crs="+init=epsg:4326")
    

## Match roads to points

def get_extended_edges(city, edges=None, batch_size=10000, cached=True, cache_result=True):

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
        with open(str(RAW_PATH/city/'extended_edges.p'), 'rb') as f:
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
        extended["hex_id"] = extended.apply(lambda row: h3.geo_to_h3(row["lat"], row["lon"], 8), axis = 1)

        if cache_result:
            with open(str(RAW_PATH/city/'extended_edges.p'), 'wb') as f:
                pickle.dump(extended, f)

            extended.drop(columns='points').to_csv(OUTPUT_PATH/city/"extended_edges.csv")
        
    return extended

def get_hex_KDTrees(city, extended=None, cached=True, cache_result=True):

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
        with open(str(RAW_PATH/city/'h3_hex_KDTree.p'), 'rb') as f:
            h3_hex = pickle.load(f)
    except:
        assert extended is not None, "extended DataFrame must be provided"
        h3_hex = pd.DataFrame(extended["hex_id"].unique(), columns=['hex_id']).set_index('hex_id', drop=False)
        h3_hex["hex_ring1"] = h3_hex["hex_id"].apply(h3.k_ring, 1, args=(1,))
        h3_hex['edges_idx_in_hex'] = h3_hex['hex_id'].apply(get_edges_idx_in_hex,args=(extended,))
        h3_hex['extended_idx_in_ring'] = h3_hex['hex_ring1'].apply(get_extended_idx_in_ring,args=(extended,))
        h3_hex['KDTree'] = h3_hex.apply(lambda row: construct_KDTree(row['hex_ring1'], extended, row['extended_idx_in_ring']),1)
        h3_hex['geometry'] = h3_hex.hex_id.apply(lambda hid: Polygon(h3.h3_to_geo_boundary(hid, geo_json=True)))
        h3_hex = gpd.GeoDataFrame(h3_hex)
        h3_hex.crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        
        if cache_result:
            with open(str(RAW_PATH/city/'h3_hex_KDTree.p'), 'wb') as f:
                pickle.dump(h3_hex, f)

    return h3_hex


def match_edges_to_points_by_hex(city, pot_holes, G_proj):
    
    edges = ox.graph_to_gdfs(G_proj, nodes=False, fill_edge_geometry=True)

    def query_KDTree(row, pot_holes_proj, pot_holes_hexs, extended):
        hex_id = row['hex_id']
        # Some hexagons in the city may not have reported pot-holes        
        if hex_id not in pot_holes_hexs:
            return np.array([], dtype=np.int64).reshape(0,5)
        pot_holes_byhex = pot_holes_proj.loc[pot_holes_proj.hex_id == hex_id]
        X = pot_holes_byhex['geometry'].apply(lambda row: row.x).values 
        Y = pot_holes_byhex['geometry'].apply(lambda row: row.y).values

        points = np.array([X, Y]).T
        _, idx = row['KDTree'].query(points, k=1)  # Returns ids of closest point
        eidx = extended.loc[row['extended_idx_in_ring']].reset_index(drop=True).loc[idx, 'index']
        ne = edges.loc[eidx, ['u', 'v','key']]
        
        # np.zeros(ne.shape[0], dtype=int)
        return np.c_[ne , pot_holes_byhex[['uuid', 'interactions']].values]    
    
    extended = get_extended_edges(city, edges)

    h3_hex = get_hex_KDTrees(city, extended)
    
    pot_holes['hex_id'] = pot_holes.apply(lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], 8), axis = 1)
    pot_holes_proj = pot_holes.to_crs(edges.crs)
    pot_holes_hexs = set(pot_holes_proj['hex_id'].values)
    
    edges_potholes = np.vstack(tuple(h3_hex.apply(query_KDTree, 1, args=(pot_holes_proj, pot_holes_hexs, extended)).values))
    edges_potholes = pd.DataFrame(edges_potholes, columns=['u','v','key','alerts_count','interactions'])

    return edges_potholes.groupby(['u','v','key']).agg({'alerts_count': 'count', 'interactions': 'sum'}).join(
        extended.groupby(['u','v','key'])['hex_id'].apply(set), how='outer').rename(columns={'hex_id':'hex_set'}).to_dict('index')



def match_roads_to_points(city, pot_holes, G_proj, by=None):
    '''
    Use projected non-simplified Graph for better accuracy in identifying pothole location
    '''
    if by == 'hex':
        return match_edges_to_points_by_hex(city, pot_holes, G_proj)

    pot_holes_proj = pot_holes.to_crs(G_proj.graph['crs'])
    
    X = pot_holes_proj['geometry'].apply(lambda row: row.x).values 
    Y = pot_holes_proj['geometry'].apply(lambda row: row.y).values 

    edges_potholes = ox.utils.get_nearest_edges(G_proj, X, Y, method='kdtree', dist=10)
    
    edges_potholes = np.c_[edges_potholes, np.zeros(edges_potholes.shape[0], dtype=int), 
                              pot_holes_proj[['uuid', 'interactions']].values]

    edges_potholes = pd.DataFrame(edges_potholes, columns=['u','v','k','alerts_count','interactions'])

    edges_potholes = edges_potholes.groupby(['u','v','k']).agg({'alerts_count': 'count', 'interactions': 'sum'}).to_dict('index')
    
    # Caching
    #pot_holes.to_csv(OUTPUT_PATH / city / 'alerts_to_segments.csv')
    #edges.to_csv(OUTPUT_PATH /  city /  'potholes_full.csv')
    
    return edges_potholes

## Basic Stats

def generate_basic_stats(city, G):
    
    edges = ox.graph_to_gdfs(G, nodes=False) 
    
    stats = {}

    stats['Total Street Length (km)'] = sum(edges['length'])/1000
    
    stats['City Total Number of Street Segments'] = len(edges)
    
    stats['Segments with potholes (#)'] = len(edges[edges['alerts_count'] > 0])

    stats['Segments with potholes (%)'] = stats['Segments with potholes (#)'] /  len(edges)  * 100

    stats['Segments Distance with potholes (km)'] = sum(edges[edges['alerts_count'] > 0]['length']) / 1000

    stats['Segments Distance with potholes (%)'] = stats['Segments Distance with potholes (km)'] / (sum(edges['length'])/1000) * 100

    stats = pd.DataFrame.from_dict(stats, 'index')
    stats.columns = [city]

    # Caching
    stats.to_csv(OUTPUT_PATH / city / 'stats.csv')

    return stats

def generate_cost_estimates(city, G, cost_per_meter, lane_size, avg_numberOf_lanes=None):
    
    edges = ox.graph_to_gdfs(G, nodes=False) 
    
    stats = {}
    
    stats['Average Reparing Cost per Square Meter (U$)'] = cost_per_meter
    
    stats['Average Street Lane Width (meters)'] = lane_size
    
    if avg_numberOf_lanes:
        stats['Average Number of Lanes in Street Segments'] = avg_numberOf_lanes
    
    stats['Estimated Fixing Cost (U$)'] = sum(edges[edges['alerts_count'] > 0]['length']) * cost_per_meter * lane_size 

    stats = pd.DataFrame.from_dict(stats, 'index')
    stats.columns = [city]
    
    # Caching
    stats.to_csv(OUTPUT_PATH / city / 'cost.csv')
    
    return stats

## Estimate Number of Lanes

def convert_lane_type(lane):

    try:
        lane = int(lane)
    except:
        lane = None
    
    return lane

def estimate_number_of_lanes(edges):

    full = edges.dropna(subset=['lanes'])

    full['lanes'] = full['lanes'].apply(convert_lane_type)

    return full.groupby(['highway']).mean()['lanes']

## Estimate Cost

def estimate_lane_cost(segment, cost_per_meter, lane_size, estimate_lanes_per_highway=None):
    
#     segment = segment.fillna(1)
        
#     if segment['lanes'] != 0:
    
#         try:
#             return float(segment['lanes']) * segment['length'] * cost_per_meter * lane_size
#         except:
#             print(type(segment['lanes']))
#             print(type(segment['length']))
            
    return lane_size * segment['length'] * cost_per_meter #* estimate_lanes_per_highway[segment['highway']]

def estimate_cost(edges, estimate_lanes_per_highway=None,
                  cost_per_meter=30, lane_size=2.8):

    edges['cost'] = edges[edges['alerts_count'] > 0].apply(estimate_lane_cost, axis=1, args=(estimate_lanes_per_highway,
                                                                                           cost_per_meter, lane_size))
    
    return edges

## Interactions

def merge_interactions_with_edges(pot_holes, edges):

    segments_interactions = pot_holes.merge((pot_holes['geometry_ids'].apply(pd.Series)
                                             .stack()
                                             .reset_index(level=1, drop=True)
                                             .to_frame('geometry_ids')), left_index=True, right_index=True, how='right')

    segments_interactions = segments_interactions.groupby('geometry_ids_y').sum()['interactions'].to_frame()
    edges = edges.merge(segments_interactions, right_index=True, left_on='geometry_id', how='left')
    edges['interactions'] = edges['interactions'].fillna(0)
    return pot_holes, edges

## Pareto

def calculate_pareto(edges, suffix=''): 

    edges['pareto'+suffix] = edges[edges['alerts_count'] > 0]['interactions'].sort_values(ascending=False).cumsum() / edges[edges['alerts_count'] > 0]['interactions'].sum() * 100

    return edges

def calculate_price_pareto(percentage, pareto, edges):
    idx = pareto[pareto <= percentage * 100].index
    selected = edges.loc[idx]
    return selected['cost'].sum(), len(selected), selected['interactions'].sum()

def summary_pareto(city, edges):

    cost_pareto = [{'percentage interactions': 0,
                    'price share (%)': 0,
                    'total cost (U$)': 0,
                    'number of segments': 0,
                    'number of interactions': 0}]
    total_price = edges['cost'].sum()
    pareto = edges['pareto']
    for i in range(1, 21):
        price, size, interactions = calculate_price_pareto(i/20, pareto, edges)
        cost_pareto.append({'percentage interactions': i*5,
                        'price share (%)': round(price / total_price * 100, 1),
                        'total cost (U$)': price,
                        'number of segments': int(size),
                        'number of interactions': int(interactions)})
    
    cost_pareto = pd.DataFrame(cost_pareto)

    # Cache
    pd.DataFrame(cost_pareto).to_csv(OUTPUT_PATH  / city / 'pareto_cost.csv')
    
    return cost_pareto

## Export Kepler

def export_kepler(city, edges):

    edges['log_interactions'] = edges['interactions'].apply(np.log10)

    a = edges[['geometry', 'alerts_count', 'name', 'interactions',
                              'log_interactions', 'cost', 'pareto']]
                              
    a = a[a['pareto'].notnull()]
    
    a.to_csv(OUTPUT_PATH / city / 'potholes_plot.csv')
    
    return a
    
    
## Save Table

def render_mpl_table(data, save_path, city, col_width=4.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.margins(0)
        ax.tick_params(which='both', left=False, bottom=False,  labelleft=False, labelbottom=False)
        # ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    
    # save figure
    ax.get_figure().savefig(OUTPUT_PATH / city / save_path, bbox_inches='tight')
    
    return fig, ax, mpl_table

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