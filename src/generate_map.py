import os
import joblib
import pandas as pd
import geopandas as gpd
import folium

print("Loading trained pipeline artifacts...")
model_path = "../models/RandomForest.joblib"
artifacts = joblib.load(model_path)

best_model = artifacts['model']
scaler = artifacts['scaler']
features = artifacts['features']

print("Loading the full Copenhagen grid...")
cph_grid_path = "../data/city_data/copenhagen_grid.geojson"
grid_wgs84 = gpd.read_file(cph_grid_path)

if grid_wgs84.crs != "EPSG:4326":
    grid_wgs84 = grid_wgs84.to_crs(epsg=4326)

df_features = pd.read_csv("../data/combined_urban_features.csv")
cph_features = df_features[df_features['city_name'].str.contains('Copenhagen', case=False, na=False)].copy()

cph_features['road_to_building_ratio'] = cph_features['total_road_length'] / (cph_features['avg_building_area'] + 1e-5)
cph_features['building_footprint_complexity'] = cph_features['shape_complexity_ratio'] * cph_features['courtyard_index']
cph_features['building_to_road_density_ratio'] = cph_features['building_density'] / (cph_features['road_density_ratio'] + 1e-5)

X_raw = cph_features[features]
X_scaled = scaler.transform(X_raw)
X_df = pd.DataFrame(X_scaled, columns=features)

print("Predicting urban eras for all valid grid cells...")
y_pred = best_model.predict(X_df)
cph_features['predicted_style'] = y_pred

grid_with_preds = grid_wgs84.merge(
    cph_features[['grid_id', 'predicted_style']], 
    left_index=True, 
    right_on='grid_id', 
    how='inner'
)

print("Initializing Folium map...")
centroid = grid_with_preds.geometry.centroid.to_crs(epsg=4326).iloc[0]
m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="cartodbpositron")

color_map = {
    'Medieval_Center': '#d95f02',   
    'XIX_Century_Blocks': '#7570b3', 
    'Modern_Urbanism': '#1b9e77'     
}

def get_style(feature):
    style_name = feature['properties']['predicted_style']
    return {
        'fillColor': color_map.get(style_name, '#888888'),
        'color': '#000000',
        'weight': 0.5,
        'fillOpacity': 0.6
    }

def get_highlight(feature):
    return {
        'weight': 2,
        'fillOpacity': 0.8
    }

folium.GeoJson(
    grid_with_preds,
    style_function=get_style,
    highlight_function=get_highlight,
    tooltip=folium.GeoJsonTooltip(
        fields=['grid_id', 'predicted_style'],
        aliases=['Grid ID:', 'Predicted Era:'],
        localize=True
    )
).add_to(m)

legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 180px; height: 110px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity: 0.85; padding: 10px;">
     <b>Urban Morphology</b><br>
     <i class="fa fa-square fa-1x" style="color:#d95f02"></i> Medieval Center<br>
     <i class="fa fa-square fa-1x" style="color:#7570b3"></i> XIX Century Blocks<br>
     <i class="fa fa-square fa-1x" style="color:#1b9e77"></i> Modern Urbanism
     </div>
     '''
m.get_root().html.add_child(folium.Element(legend_html))

output_dir = "../plots/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "copenhagen_prediction_map.html")
m.save(output_path)
print(f"Success! Interactive map saved to: {output_path}")