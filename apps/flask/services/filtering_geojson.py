import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import json

# Load the full Aceh GeoJSON
aceh_file = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Repository/kkp-only/tumbuh-baik/apps/flask/data/geojson/gadm41_IDN_3.json"
aceh = gpd.read_file(aceh_file)
aceh = aceh[aceh["NAME_1"] == "Aceh"] 

print(f"Total kecamatan in Aceh: {len(aceh)}")

# NASA POWER dataset coordinates (11 locations - Removed Pasie Raya)
nasa_coordinates = {
    "Lhoksukon": (5.0323933, 97.31531),        # Aceh Utara
    "Juli": (5.1872539, 96.6991391),           # Bireuen
    "Kota Juang": (5.2067313, 96.711344),      # Bireuen
    "Indrapuri": (5.4117925, 95.4418927),      # Aceh Besar
    "Montasik": (5.465827, 95.4340119),        # Aceh Besar
    "Darussalam": (5.5859711, 95.3913794),     # Aceh Besar
    "Jaya": (5.0804551, 95.3515282),           # Aceh Jaya (Jaya/Lamno)
    "Setia Bakti": (4.7639286, 95.5762011),    # Aceh Jaya
    "Teunom": (4.451494, 95.8573616),          # Aceh Jaya
    "Pidie": (5.3580667, 95.9413261),          # Pidie
    "Indrajaya": (5.3108487, 95.9459862)       # Pidie
}

print(f"NASA locations to match: {len(nasa_coordinates)}")

# SPATIAL JOIN: Find polygons containing NASA points
print("Performing spatial join...")

matched_rows = []
not_found = []

for nasa_name, (lat, lng) in nasa_coordinates.items():
    nasa_point = Point(lng, lat)
    containing_polygons = aceh[aceh.geometry.contains(nasa_point)]
    
    if len(containing_polygons) == 1:
        row = containing_polygons.iloc[0].copy()
        row["nasa_match"] = nasa_name
        row["nasa_lat"] = lat
        row["nasa_lng"] = lng
        matched_rows.append(row)
        print(f"Matched: {nasa_name} -> {row['NAME_3']}")
        
    elif len(containing_polygons) > 1:
        print(f"Warning: {nasa_name} found in multiple polygons: {containing_polygons['NAME_3'].tolist()}")
        row = containing_polygons.iloc[0].copy()
        row["nasa_match"] = nasa_name
        row["nasa_lat"] = lat
        row["nasa_lng"] = lng
        matched_rows.append(row)
        
    else:
        print(f"Error: {nasa_name} at ({lat}, {lng}) not found in any polygon")
        not_found.append(nasa_name)

# Create GeoDataFrame from matched rows
if matched_rows:
    matched_kecamatan = gpd.GeoDataFrame(matched_rows, crs=aceh.crs)
    print(f"Successfully matched {len(matched_kecamatan)} of {len(nasa_coordinates)} NASA locations")
else:
    print("No matches found!")
    exit(1)

# Show missing matches
if not_found:
    print(f"NASA locations not found: {not_found}")

# Calculate centroids with proper CRS handling
print("Calculating centroids...")
matched_kecamatan_utm = matched_kecamatan.to_crs("EPSG:32647")
centroids_utm = matched_kecamatan_utm.geometry.centroid
centroids_geo = centroids_utm.to_crs("EPSG:4326")
matched_kecamatan["centroid_lat"] = centroids_geo.y
matched_kecamatan["centroid_lng"] = centroids_geo.x

# Validate coordinates
within_count = 0
for idx, row in matched_kecamatan.iterrows():
    nasa_point = Point(row["nasa_lng"], row["nasa_lat"])
    is_within = row.geometry.contains(nasa_point)
    if is_within:
        within_count += 1

print(f"Spatial validation: {within_count}/{len(matched_kecamatan)} points within polygons")

# Clean up columns for final export
final_columns = [
    "GID_3", "NAME_1", "NAME_2", "NAME_3", "TYPE_3", 
    "nasa_match", "nasa_lat", "nasa_lng", 
    "centroid_lat", "centroid_lng", "geometry"
]

filtered_geojson = matched_kecamatan[final_columns].copy()

# Ensure proper CRS for export
if filtered_geojson.crs is None:
    filtered_geojson.crs = "EPSG:4326"

# Export filtered GeoJSON
output_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Repository/kkp-only/tumbuh-baik/apps/flask/data/geojson/aceh_nasa_kecamatan.geojson"
filtered_geojson.to_file(output_path, driver="GeoJSON")

print(f"Filtered GeoJSON saved to: {output_path}")

# Create summary for Flask integration
summary = {
    "total_kecamatan": len(filtered_geojson),
    "nasa_locations_count": len(nasa_coordinates),
    "kabupaten_distribution": filtered_geojson["NAME_2"].value_counts().to_dict(),
    "matched_locations": filtered_geojson[["NAME_3", "nasa_match", "NAME_2", 
                                         "nasa_lat", "nasa_lng", "centroid_lat", "centroid_lng"]].to_dict("records"),
    "crs_info": {
        "output_crs": "EPSG:4326",
        "projection_used": "EPSG:32647 (UTM Zone 47N)"
    }
}

# Save summary as JSON  
summary_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Repository/kkp-only/tumbuh-baik/apps/flask/data/geojson/nasa_kecamatan_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"Summary saved to: {summary_path}")

# Final statistics
print(f"Final dataset: {len(filtered_geojson)} kecamatan")
print(f"Kabupaten covered: {len(filtered_geojson['NAME_2'].unique())}")

# Verify no duplicates
gid_counts = filtered_geojson['GID_3'].value_counts()
duplicates = gid_counts[gid_counts > 1]

if len(duplicates) > 0:
    print(f"Warning: Found duplicate GID_3 values")
else:
    print("No duplicate GID_3 values found")