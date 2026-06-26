import osmnx as ox
import geopandas as gpd
import pandas as pd
from building_features import extract_building_features
import json
from road_features import extract_road_features
from shapely.geometry import Point

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

def get_city_zones_by_buffer(zones_config):
    loaded_zones = {}
    for era_name, district_name in zones_config.items():
        print(f"  Creating zone for {era_name} ({district_name})...")
        try:
            # 1. Pobiera tylko współrzędne (X, Y) środka dzielnicy
            lat, lon = ox.geocode(district_name)
            
            # 2. Tworzy punkt i rzutuje go na metry (EPSG:32632)
            point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=32632)
            point_geom = point_gdf.geometry.iloc[0]
            
            # 3. Tworzy idealny okrąg o promieniu 1000 metrów wokół tego punktu
            loaded_zones[era_name] = point_geom.buffer(1000)
            
        except Exception as e:
            print(f"  Critical Error: Could not resolve location {district_name}: {e}")
                
    return loaded_zones

def run_pipeline_for_city(place_name, grid_geojson_path, zones_config):
    print(f"\n--- Processing: {place_name} ---")
    spatial_grid_m = gpd.read_file(grid_geojson_path).to_crs(epsg=32632)
    
    # Load specific zones for this city
    zones = get_city_zones_by_buffer(city_config["zones"])
    
    print("Downloading raw OSM data for feature extraction...")
    try:
        buildings = ox.features_from_place(place_name, tags={"building": True})
        buildings_m = buildings[['geometry']].dropna().to_crs(epsg=32632)
        
        graph = ox.graph_from_place(place_name, network_type="all")
        roads = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        roads_m = roads[['geometry']].dropna().to_crs(epsg=32632)
    except Exception as e:
        print(f"Error loading OSM data for {place_name}: {e}")
        return []
    
    print(f"Starting processing loop for {len(spatial_grid_m)} cells...")
    city_features = []
    
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
        
        # 1. Extract building features
        b_features = extract_building_features(b_in_cell, cell.geometry)
        r_features = extract_road_features(r_in_cell)
        feature_row = {
            'city_name': place_name,
            'grid_id': idx,
            'target_style': target_style, 
            **b_features, # Unpacking all building columns directly into the row
            **r_features,
        }
        
        city_features.append(feature_row)
        
    return city_features

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
            output_csv_path="../data/combined_urban_features.csv"
        )