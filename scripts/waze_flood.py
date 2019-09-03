import numpy as np
import pandas as pd
import geopandas as gpd
import pytz
from shapely.geometry import Point
import sqlalchemy as sa
from pyathena import connect
import osmnx as ox
import sys
abs_path = '/home/master/cts/cities/waze-tools'
sys.path.append(abs_path)
import wazetools.data.data_transform as dt
import wazetools.data.join as join
from datetime import timedelta
from geopy.distance import great_circle
import sys
from pathlib import Path 

current_path = Path().resolve()
RAW_PATH = current_path.parent / 'data' / 'raw'
OUTPUT_PATH = current_path.parent / 'data' / 'output' 



conn = connect(s3_staging_dir='s3://aws-athena-queries-result-east-2/join-pems-waze',
               region_name='us-east-2')

con = sa.create_engine(open('../../key_redshift.txt', 'r').read())

## Get Data

def download_accidents_data(city, cities, unique=True):
    
    query = f"""select uuid, pub_utc_date, location_x, location_y, subtype,
                    reliability, confidence, thumbs_up, jam_uuid, report_by_municipality_user
                    from waze.alerts
                    where city = '{city}' AND type = 'ACCIDENT'"""
    if unique:
        query = f"""select DISTINCT uuid, MIN(pub_utc_date) as start_time, MAX(pub_utc_date) as end_time, 
                        AVG(location_x) as location_x, AVG(location_y) as location_y, MAX(subtype) as subtype,
                        MIN(reliability) as reliability_min, MAX(reliability) as reliability_max,
                        MIN(confidence) as confidence_min, MAX(confidence) as confidence_max,
                        MAX(thumbs_up) as thumbs_up
                        from waze.alerts
                        where city = '{city}' AND type = 'ACCIDENT'
                        group by uuid"""
    
    cities[city]['accidents'] = pd.read_sql_query(query, con=con)
    
    cities[city]['accidents']['start_time'] = cities[city]['accidents']['start_time'].apply(lambda x: x.replace(tzinfo=pytz.utc)
                                                                                                    .astimezone(cities[city]['timezone'])
                                                                                                    .replace(tzinfo=None))
    
    cities[city]['accidents']['end_time'] = cities[city]['accidents']['end_time'].apply(lambda x: x.replace(tzinfo=pytz.utc)
                                                                                                    .astimezone(cities[city]['timezone'])
                                                                                                    .replace(tzinfo=None))
    return cities

def download_osm_graph(city, cities):
    polygon = gpd.read_file(cities[city]['polygon_path'])['geometry'].values[0]
    cities[city]['G'] = ox.graph_from_polygon(polygon, network_type='drive')
    cities[city].update(dt.get_osm(city, cities[city]['G']))
    return cities

def get_cities_metadata():
    cities = pd.read_sql_table('cities_stats', schema='waze', con=con).set_index('city_as_waze')
    cities['timezone'] = cities['timezone'].map(lambda x: pytz.timezone(x))
    return cities.to_dict('index')

def get_timerange(df):
    start = df['start_time'].min()
    end = df['end_time'].max()
    return pd.Series(data=[start, end])

## Treat data

def treat_alert_points(df):
    df['point'] = df.apply(lambda x: Point(x['location_x'], x['location_y']), axis=1)
    

# Clustering

def clustering(data, eps_spatial, eps_temporal):
    """
    Agrega alertas que tem a mesma origem. 

    Input: 
    data :: pd.Dataframe com colunas --> location_x, location_y, initial_time, end_time, uuid (grouped by uuid)
    eps_temporal :: float que representa a altura do cilindro em segundos
    eps_spatial :: float que representa o raio do cilindro em metros

    output:
        clusters ::  list of dict. Cada dict com as keys
              'cluster_id': int or string,
              'alerts_uuid': list of uuids,
              'initial_time' : timestamp,
              'end_time' : timestamp,
              'total_time' : timestamp,
              'estimated_center' : {'latitude': float, 'longitude': float},
              'estimated_polygon': Polygon structure
      """
    # eps_temporal: segundos
    # eps_spatial: metros
    
    aggregation = {'uuid': 'count', 'start_time': min, 'end_time' : max, 'location_x': 'mean', 'location_y': 'mean', 
               'reliability_min': min, 'reliability_max': max, 'confidence_min': min, 'confidence_max': max, 'thumbs_up': sum}
    
    data_gb = st_dbscan(data, eps_spatial, eps_temporal, 0).groupby('cluster')
    clusters = data_gb.agg(aggregation).rename(columns={'uuid':"uuid count"}).to_dict('index')
    for cluster in clusters:
        clusters[cluster]['uuid list'] = data_gb.get_group(cluster)['uuid'].tolist()
    return clusters

def st_dbscan(data, eps_spatial, eps_temporal, min_neighbors):
    data = data.copy()
    cluster = 0
    unmarked = 888888
    noise = -1
    stack = []
    data['cluster'] = unmarked
    for index, point in data.iterrows():
        if data.loc[index]['cluster'] == unmarked:
            neighborhood = retrieve_neighbors(index, data, eps_spatial, eps_temporal)

            if len(neighborhood) < min_neighbors:
                data.at[index, 'cluster'] = noise
            else:
                cluster += 1
                data.at[index, 'cluster'] = cluster
                for neig_index in neighborhood:
                    data.at[neig_index, 'cluster'] = cluster
                    stack.append(neig_index)  # append neighborhood to stack
                # find new neighbors from core point neighborhood
                while len(stack) > 0:
                    current_point_index = stack.pop()
                    new_neighborhood = retrieve_neighbors(current_point_index, data, eps_spatial, eps_temporal)

                    # current_point is a new core
                    if len(new_neighborhood) >= min_neighbors:
                        for neig_index in new_neighborhood:
                            neig_cluster = data.loc[neig_index]['cluster']
                            if any([neig_cluster == noise, neig_cluster == unmarked]):
                                data.at[neig_index, 'cluster'] = cluster
                                stack.append(neig_index)
    return data

def retrieve_neighbors(index_center, data, eps_spatial, eps_temporal):
    neigborhood = []
    center_point = data.loc[index_center]

    # filter by time
    min_time = center_point['start_time'] - timedelta(seconds=eps_temporal)
    max_time = center_point['end_time'] + timedelta(seconds=eps_temporal)
    data = data[((data['start_time'] >= min_time) & (data['start_time'] <= max_time)) | 
               ((data['end_time'] >= min_time) & (data['end_time'] <= max_time))]

    # filter by distance
    for index, point in data.iterrows():
        if index != index_center:
            distance = great_circle(
                (center_point['location_y'], center_point['location_x']),
                (point['location_y'], point['location_x'])).meters
            if distance <= eps_spatial:
                neigborhood.append(index)

    return neigborhood

## Match roads to points

def match_pothole_to_road(pothole, edges):
    
    i = 0
    while i < 5:
        
        pothole_box = join.bbox_from_point(pothole, 0.0001 * i + 0.0001)
    
        intersects = edges['geometry'].apply(lambda x: x.intersects(pothole_box))
        
        if sum(intersects):
            return edges[intersects]['geometry_id'].values
        
        else:
            i = i + 1

def match_roads_to_points(city, cities):

    cities[city]['potholes_unique']['geometry_ids'] = \
        cities[city]['potholes_unique']['point'].apply(
            match_pothole_to_road, args=(cities[city]['edges'],))

    geometry_ids = []
    for geometry in cities[city]['potholes_unique']['geometry_ids']:
        if geometry is not None:
            geometry_ids.extend(geometry)

    cities[city]['edges']['pothole_exists'] = \
        cities[city]['edges']['geometry_id'].isin(geometry_ids)

    # Caching
    cities[city]['potholes_unique'].to_csv(OUTPUT_PATH / city / 'alerts_to_segments.csv')
    cities[city]['edges'].to_csv(OUTPUT_PATH /  city /  'potholes_full.csv')

    return cities

## Basic Stats

def generate_basic_stats(city, cities):

    stats = {}

    stats['Segments Total (#)'] = \
    len(cities[city]['edges'].drop_duplicates(subset=['geometry_id'])) 

    stats['Segments with potholes (#)'] = \
    sum(cities[city]['edges'].drop_duplicates(subset=['geometry_id'])['pothole_exists']) 

    stats['Segments with potholes (%)'] = \
    sum(cities[city]['edges'].drop_duplicates(subset=['geometry_id'])['pothole_exists']) / len(cities[city]['edges'].drop_duplicates(subset=['geometry_id'])) * 100

    stats['Segments Distance Total (km)'] = \
    sum(cities[city]['edges']['length'])/1000

    stats['Segments Distance with potholes (km)'] = \
    sum(cities[city]['edges'][cities[city]['edges']['pothole_exists']]['length']) / 1000

    stats['Segments Distance with potholes (%)'] = \
    stats['Segments Distance with potholes (km)'] / stats['Segments Distance Total (km)'] * 100

    stats = pd.DataFrame.from_dict(stats, 'index')
    stats.columns = [city]

    # Caching
    stats.to_csv(OUTPUT_PATH / city / 'stats.csv')

    return stats

## Estimate Number of Lanes

def estimate_number_of_lanes(city, cities):

    full = cities[city]['edges'].dropna(subset=['lanes'])

    full['lanes'] = full['lanes'].apply(int)

    return full.groupby(['highway']).mean()['lanes']

## Estimate Cost

def estimate_lane_cost(segment, estimate_lanes_per_highway,
                        cost_per_meter, lane_size):
    
    segment = segment.fillna(0)
        
    if segment['lanes'] != 0:
    
        try:
            return float(segment['lanes']) * segment['length'] * cost_per_meter
        except:
            print(type(segment['lanes']))
            print(type(segment['length']))
            
    # print(estimate_lanes_per_highway[segment['highway']])
    # print(segment['length'])
    # print()
    return estimate_lanes_per_highway[segment['highway']] * segment['length'] * cost_per_meter

def estimate_cost(city, cities, estimate_lanes_per_highway,
                  cost_per_meter=30, lane_size=2.8):

    cities[city]['edges']['cost'] = \
        cities[city]['edges'][cities[city]['edges']['pothole_exists']].apply(
            estimate_lane_cost, axis=1, args=(estimate_lanes_per_highway,
                                             cost_per_meter, lane_size))
    
    return cities

## Interactions

def merge_interactions_with_edges(city, cities):

    cities[city]['potholes_unique'] = \
        cities[city]['potholes_unique'][['uuid', 'location_x', 'location_y', 'point', 'geometry_ids']]

    cities[city]['edges'] = cities[city]['edges'][['access', 'bridge', 'geometry', 'highway', 'junction', 'k', 'lanes',
       'length', 'maxspeed', 'name', 'oneway', 'osmid', 'ref', 
       'tunnel', 'u', 'v', 'width', 'geometry_id', 'pothole_exists', 'cost',]]


    cities[city]['potholes_unique'] = cities[city]['potholes_unique'].merge(cities[city]['potholes_interactions'], on='uuid', how='left')

    uuid_to_segments = cities[city]['potholes_unique'].merge(
                (cities[city]['potholes_unique']['geometry_ids'].apply(pd.Series)
                .stack()
                .reset_index(level=1, drop=True)
                .to_frame('geometry_ids')), left_index=True, right_index=True, how='right')[['uuid', 'geometry_ids_y']]

    segments_interactions = cities[city]['potholes_unique'][['uuid', 'interactions']].merge(uuid_to_segments, on='uuid')
    segments_interactions['interactions'] = segments_interactions['interactions'] + 1 
    segments_interactions = segments_interactions.groupby('geometry_ids_y').sum()['interactions'].to_frame()
    cities[city]['edges'] = cities[city]['edges'].merge(segments_interactions, right_index=True, left_on='geometry_id', how='left')
    cities[city]['edges']['interactions'] = cities[city]['edges']['interactions'].fillna(0)
    return cities

## Pareto

def calculate_pareto(city, cities):

    cities[city]['edges'] = cities[city]['edges'].drop_duplicates('geometry_id')
    cities[city]['edges']['pareto'] = cities[city]['edges'][cities[city]['edges']['pothole_exists']]['interactions'].sort_values(ascending=False).cumsum() / cities[city]['edges'][cities[city]['edges']['pothole_exists']]['interactions'].sum() * 100

    return cities

def calculate_price_pareto(percentage, pareto, cities, city):
    idx = pareto[pareto <= percentage * 100].index
    selected = cities[city]['edges'].loc[idx]
    return selected['cost'].sum(), len(selected), selected['interactions'].sum()

def summary_pareto(city, cities):

    cost_pareto = []
    total_price = cities[city]['edges']['cost'].sum()
    pareto = cities[city]['edges']['pareto']
    for i in range(1, 11):
        price, size, interactions = calculate_price_pareto(i/10, pareto, cities, city)
        cost_pareto.append({'percentage interactions': i*10,
                        'price share (%)': round(price / total_price * 100, 1),
                        'number of roads': size,
                        'number of interactions': interactions})
    
    cost_pareto = pd.DataFrame(cost_pareto)

    # Cache
    pd.DataFrame(cost_pareto).to_csv(OUTPUT_PATH  / city / 'pareto_cost.csv')
    
    return cost_pareto

## Export Kepler

def export_kepler(city, cities):

    cities[city]['edges']['log_interactions'] = cities[city]['edges']['interactions'].apply(np.log10)

    cities[city]['edges'][['geometry', 'pothole_exists', 'name', 'interactions',
                              'log_interactions', 'cost', 'pareto']].to_csv(OUTPUT_PATH / city / 'potholes_plot.csv')