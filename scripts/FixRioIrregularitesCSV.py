# This script replace the original list of coordinates by a LineString object 
# that can be used in Kepler.gl visualizations

import pandas as pd
import geopandas as gpd
import ast
import re

from shapely.geometry import Point, Polygon, LineString

from pathlib import Path 
current_path = Path().resolve()
DATA_PATH = current_path.parent / 'data'

rio = pd.read_csv(DATA_PATH / "Rio-Worst-Day-irreg.csv")

rio['line'] = rio['line'].apply(lambda x: x.replace('{','(').replace('}',')').replace('x=', '').replace('y=', ''))

rio['line'] = rio['line'].apply(ast.literal_eval)

rio['geometry'] = rio['line'].apply(LineString)

rio.to_csv(DATA_PATH / "Rio-Worst-Day-irreg.csv")