import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import json

# Load the full Aceh GeoJSON
aceh_file = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Repository/kkp-only/tumbuh-baik/apps/flask/data/geojson/gadm41_IDN_3.json"
aceh = gpd.read_file(aceh_file)
aceh = aceh[aceh["NAME_1"] == "Aceh"] 

print(f"📊 Total kecamatan in Aceh: {len(aceh)}")
print(f"🗺️ Original CRS: {aceh.crs}")

# NASA POWER dataset coordinates (11 locations - REMOVED Pasie Raya)
nasa_coordinates = {
    "Indrapuri": (5.4218918, 95.4463322),      # Aceh Besar
    "Montasik": (5.4803, 95.4594),             # Aceh Besar  
    "Darussalam": (5.5945451, 95.4201377),     # Aceh Besar
    "Jaya": (5.1526721, 95.1509617),           # Aceh Jaya (Jaya/Lamno)
    "Setia Bakti": (4.8333007, 95.4867971),    # Aceh Jaya
    "Teunom": (4.4999535, 95.7700208),         # Aceh Jaya
    "Pidie": (5.3759998, 95.9148038),          # Pidie
    "Indrajaya": (5.3114261, 95.8978653),      # Pidie
    "Lhoksukon": (5.051701, 97.318123),        # Aceh Utara
    "Juli": (5.0735373, 96.5879472),           # Bireuen
    "Kota Juang": (5.190849, 96.6728368)       # Bireuen
}

print(f"🎯 NASA locations to match: {len(nasa_coordinates)}")

# SPATIAL JOIN: Find polygons containing NASA points
print("🎯 Performing spatial join using point-in-polygon...")

matched_rows = []
not_found = []

for nasa_name, (lat, lng) in nasa_coordinates.items():
    # Create NASA point (note: Point takes (lng, lat) order)
    nasa_point = Point(lng, lat)
    
    # Find kecamatan polygon that contains this NASA point
    containing_polygons = aceh[aceh.geometry.contains(nasa_point)]
    
    if len(containing_polygons) == 1:
        # Perfect match: exactly one polygon contains the point
        row = containing_polygons.iloc[0].copy()
        row["nasa_match"] = nasa_name
        row["nasa_lat"] = lat
        row["nasa_lng"] = lng
        matched_rows.append(row)
        print(f"  ✅ {nasa_name} → {row['NAME_3']} (Kabupaten: {row['NAME_2']})")
        
    elif len(containing_polygons) > 1:
        # Multiple polygons contain the point (rare edge case)
        print(f"  ⚠️ {nasa_name} found in multiple polygons: {containing_polygons['NAME_3'].tolist()}")
        # Take the first one
        row = containing_polygons.iloc[0].copy()
        row["nasa_match"] = nasa_name
        row["nasa_lat"] = lat
        row["nasa_lng"] = lng
        matched_rows.append(row)
        
    else:
        # No polygon contains the point
        print(f"  ❌ {nasa_name} at ({lat}, {lng}) not found in any polygon")
        not_found.append(nasa_name)

# Create GeoDataFrame from matched rows
if matched_rows:
    matched_kecamatan = gpd.GeoDataFrame(matched_rows, crs=aceh.crs)
    print(f"\n✅ Successfully matched {len(matched_kecamatan)} of {len(nasa_coordinates)} NASA locations")
else:
    print("❌ No matches found!")
    exit(1)

# Show any missing matches
if not_found:
    print(f"\n⚠️ NASA locations not found in polygons: {not_found}")
    
    # Try to find nearest polygons for debugging
    for missing_name in not_found:
        lat, lng = nasa_coordinates[missing_name]
        nasa_point = Point(lng, lat)
        
        # Calculate distances to all polygons
        aceh_copy = aceh.copy()
        aceh_copy['distance'] = aceh_copy.geometry.distance(nasa_point)
        nearest = aceh_copy.loc[aceh_copy['distance'].idxmin()]
        
        print(f"  📍 {missing_name} nearest to: {nearest['NAME_3']} (Kabupaten: {nearest['NAME_2']}) "
              f"- Distance: {nearest['distance']:.6f} degrees")

# Calculate centroids with proper CRS handling
print("\n🗺️ Calculating centroids with proper projection...")

# Project to UTM Zone 47N (appropriate for Aceh, Indonesia)
matched_kecamatan_utm = matched_kecamatan.to_crs("EPSG:32647")  # UTM Zone 47N

# Calculate centroids in projected coordinate system
centroids_utm = matched_kecamatan_utm.geometry.centroid

# Convert centroids back to geographic coordinates (WGS84)
centroids_geo = centroids_utm.to_crs("EPSG:4326")  # WGS84

# Extract lat/lng from projected centroids
matched_kecamatan["centroid_lat"] = centroids_geo.y
matched_kecamatan["centroid_lng"] = centroids_geo.x

# Validate coordinates - should all be within polygon now
print("\n🎯 Coordinate validation:")
within_count = 0
for idx, row in matched_kecamatan.iterrows():
    nasa_point = Point(row["nasa_lng"], row["nasa_lat"])
    is_within = row.geometry.contains(nasa_point)
    
    # Calculate distance between NASA point and centroid
    nasa_coord = (row["nasa_lat"], row["nasa_lng"])
    centroid_coord = (row["centroid_lat"], row["centroid_lng"])
    
    # Simple distance calculation (approximate)
    lat_diff = abs(nasa_coord[0] - centroid_coord[0])
    lng_diff = abs(nasa_coord[1] - centroid_coord[1])
    distance_km = ((lat_diff ** 2 + lng_diff ** 2) ** 0.5) * 111  # Rough km conversion
    
    status = "✅" if is_within else "❌"
    if is_within:
        within_count += 1
    
    print(f"  {row['nasa_match']}: {status} "
          f"NASA({row['nasa_lat']:.4f}, {row['nasa_lng']:.4f}) "
          f"Centroid({row['centroid_lat']:.4f}, {row['centroid_lng']:.4f}) "
          f"[~{distance_km:.1f}km]")

print(f"\n📊 Spatial validation: {within_count}/{len(matched_kecamatan)} points within polygons")

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

print(f"\n💾 Filtered GeoJSON saved to: {output_path}")
print(f"📊 Final dataset: {len(filtered_geojson)} kecamatan")

# Create summary for Flask integration
summary = {
    "total_kecamatan": len(filtered_geojson),
    "nasa_locations_count": len(nasa_coordinates),
    "kabupaten_distribution": filtered_geojson["NAME_2"].value_counts().to_dict(),
    "coordinate_validation": {
        "within_polygon": within_count,
        "total_checked": len(matched_kecamatan)
    },
    "matched_locations": filtered_geojson[["NAME_3", "nasa_match", "NAME_2", 
                                         "nasa_lat", "nasa_lng", "centroid_lat", "centroid_lng"]].to_dict("records"),
    "nasa_locations_not_found": not_found,
    "removed_invalid_locations": ["Pasie Raya"],  # Track what was removed
    "crs_info": {
        "output_crs": "EPSG:4326",
        "projection_used": "EPSG:32647 (UTM Zone 47N)"
    }
}

# Save summary as JSON  
summary_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Repository/kkp-only/tumbuh-baik/apps/flask/data/geojson/nasa_kecamatan_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"📋 Summary saved to: {summary_path}")

# Show final statistics
print(f"\n📈 Final Statistics:")
print(f"  - NASA locations to process: {len(nasa_coordinates)}")
print(f"  - Total kecamatan matched: {len(filtered_geojson)}")
print(f"  - Kabupaten covered: {len(filtered_geojson['NAME_2'].unique())}")
print(f"  - NASA locations found: {len(matched_kecamatan)}/{len(nasa_coordinates)}")
print(f"  - Spatial accuracy: {within_count}/{len(matched_kecamatan)} points within polygons")
print(f"  - Removed invalid locations: ['Pasie Raya']")

# Verify no duplicates in final output
gid_counts = filtered_geojson['GID_3'].value_counts()
duplicates = gid_counts[gid_counts > 1]

if len(duplicates) > 0:
    print(f"\n⚠️ WARNING: Found duplicate GID_3 values:")
    for gid, count in duplicates.items():
        print(f"  {gid}: {count} occurrences")
else:
    print(f"\n✅ No duplicate GID_3 values found - each kecamatan appears exactly once")