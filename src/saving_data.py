import osmnx as ox
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import json

def load_pipeline_config(config_path="../data/config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_urban_grid(place_name, square_size=400, output_filename=None, plot_result=True):
    ox.settings.use_cache = True
    
    print(f"Loading boundary for: {place_name}...")
    try:
        boundary = ox.geocode_to_gdf(place_name)
    except Exception as e:
        print(f"Error locating place '{place_name}': {e}")
        return None
        
    boundary_m = boundary.to_crs(epsg=32632)
    xmin, ymin, xmax, ymax = boundary_m.total_bounds
    
    print(f"Generating grid with square size {square_size}m...")
    grid_boxes = []
    for x in np.arange(xmin, xmax, square_size):
        for y in np.arange(ymin, ymax, square_size):
            grid_boxes.append(box(x, y, x + square_size, y + square_size))
            
    spatial_grid_m = gpd.GeoDataFrame(geometry=grid_boxes, crs="EPSG:32632")
    spatial_grid_m = gpd.sjoin(spatial_grid_m, boundary_m, how="inner", predicate="intersects")
    spatial_grid_m = spatial_grid_m[['geometry']].reset_index(drop=True)
    
    print(f"Generated {len(spatial_grid_m)} squares.")
    
    if output_filename:
        print(f"Saving grid to {output_filename}...")
        spatial_grid_m.to_file(output_filename, driver="GeoJSON")
        print("Saved successfully!")
        
    if plot_result:
        fig, ax = plt.subplots(figsize=(12, 12))
        boundary_m.plot(ax=ax, color="#EAEAEA", edgecolor="black", linewidth=2)
        spatial_grid_m.plot(ax=ax, facecolor="none", edgecolor="#3498DB", linewidth=0.5, alpha=0.7)
        ax.set_title(f"{place_name} divided into squares {square_size}x{square_size}m (N={len(spatial_grid_m)})")
        ax.axis("off")
        plt.show()
        
    return spatial_grid_m

if __name__ == "__main__":
    config = load_pipeline_config("../data/config.json")
    
    square_size = config.get("square_size_meters", 400)
    
    for city_name, city_config in config["cities"].items():
        print(f"\n--- Starting Grid Generation for: {city_name} ---")
        generate_urban_grid(
            place_name=city_name, 
            square_size=square_size, 
            output_filename=city_config["grid_path"], 
            plot_result=True
        )