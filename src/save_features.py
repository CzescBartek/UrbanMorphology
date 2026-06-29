import osmnx as ox
import geopandas as gpd
import pandas as pd
from building_features import extract_building_features
import json
from road_features import extract_road_features
from shapely.geometry import Point
from pyproj import CRS
import os

def load_pipeline_config(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_features_to_csv(features_dict_list, output_csv_path):
    print("Converting features to DataFrame...")
    df_features = pd.DataFrame(features_dict_list)
    print(f"Saving features to {output_csv_path}...")
    df_features.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df_features)} rows successfully.")
    return df_features

def get_city_zones_by_buffer(zones_config, target_crs):
    loaded_zones = {}
    for era_name, district_name in zones_config.items():
        print(f"  Creating zone for {era_name} ({district_name})...")
        try:
            lat, lon = ox.geocode(district_name)
            
            # Dynamicznie rzutujemy punkt na CRS przekazany z siatki miasta (target_crs)
            point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(target_crs)
            point_geom = point_gdf.geometry.iloc[0]
            
            loaded_zones[era_name] = point_geom.buffer(1000)
            
        except Exception as e:
            print(f"  Critical Error: Could not resolve location {district_name}: {e}")
                
    return loaded_zones
def run_pipeline_for_city(place_name, grid_geojson_path, zones_config):
    print(f"\n--- Processing: {place_name} ---")
    if not os.path.exists(grid_geojson_path):
        print(f"Error: Grid file missing at {grid_geojson_path}")
        return []

    spatial_grid = gpd.read_file(grid_geojson_path)
    local_utm_crs = spatial_grid.estimate_utm_crs()
    spatial_grid_m = spatial_grid.to_crs(local_utm_crs)
    
    zones = get_city_zones_by_buffer(zones_config, local_utm_crs)
    
    print("Downloading raw OSM data for feature extraction...")
    try:
        buildings = ox.features_from_place(place_name, tags={"building": True})
        buildings_m = buildings[['geometry']].dropna().to_crs(local_utm_crs)
        
        graph = ox.graph_from_place(place_name, network_type="all")
        roads = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        roads_m = roads[['geometry']].dropna().to_crs(local_utm_crs)
    except Exception as e:
        print(f"Error loading OSM data for {place_name}: {e}")
        return []
    
    print(f"Starting processing loop for {len(spatial_grid_m)} cells...")
    city_rows = []
    
    for idx, cell in spatial_grid_m.iterrows():
        target_style = "Unknown"
        max_intersection = 0.0
        
        for zone_name, zone_geometry in zones.items():
            if cell.geometry.intersects(zone_geometry):
                intersection_area = cell.geometry.intersection(zone_geometry).area
                if intersection_area > max_intersection:
                    max_intersection = intersection_area
                    target_style = zone_name
        
        if target_style == "Unknown" or max_intersection < (cell.geometry.area * 0.3):
            continue
            
        b_in_cell = buildings_m[buildings_m.intersects(cell.geometry)]
        r_in_cell = roads_m[roads_m.intersects(cell.geometry)]
        
        b_features = extract_building_features(b_in_cell, cell.geometry)
        r_features = extract_road_features(r_in_cell)
        
        feature_row = {
            'city_name': place_name,
            'grid_id': idx,
            'target_style': target_style, 
            'geometry': cell.geometry,
            **b_features, 
            **r_features,
        }
        city_rows.append(feature_row)
        
    if not city_rows:
        return []
        
    city_gdf = gpd.GeoDataFrame(city_rows, crs=local_utm_crs)
    
    print("Calculating spatial lag features (Context)...")
    centroids = city_gdf.geometry.centroid
    spatial_lags = []
    
    for i, centroid in enumerate(centroids):
        distances = centroids.distance(centroid)
        neighbors = city_gdf[(distances <= 600) & (distances > 0)]
        
        if not neighbors.empty:
            spatial_lags.append(neighbors['building_density'].mean())
        else:
            spatial_lags.append(city_gdf['building_density'].iloc[i])
            
    city_gdf['building_density_spatial_lag'] = spatial_lags
    city_df = pd.DataFrame(city_gdf.drop(columns=['geometry']))
    
    print(f"--> Finished {place_name}. Generated {len(city_df)} valid cells with spatial context.")
    return city_df.to_dict(orient='records')

if __name__ == "__main__":
    config = load_pipeline_config("../data/config.json")
    all_calculated_features = []
    
    for city_name, city_config in config['cities'].items():
        city_data = run_pipeline_for_city(
            place_name=city_name, 
            grid_geojson_path=city_config["grid_path"],
            zones_config=city_config["zones"]
        )
        all_calculated_features.extend(city_data)
        
    if all_calculated_features:
        save_features_to_csv(
            features_dict_list=all_calculated_features, 
            output_csv_path=config["output_csv_path"]
        )
    else:
        print("\nCritical Error: No features generated for any city. Check if geojson files exist.")