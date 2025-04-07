import folium
from shapely.geometry import Point
import geopandas as gpd

# Create map
fire_map = folium.Map(location=[37, -120], zoom_start=6)

# Simulated predicted fire locations (use actual prediction data)
predicted_fire_locations = [(37.5, -120.5), (37.6, -120.6), (37.7, -120.4)]

# Add predicted fire locations as markers
for lat, lon in predicted_fire_locations:
    folium.Marker([lat, lon], popup="Predicted Fire Location").add_to(fire_map)

# Simulated evacuation routes (example)
evacuation_routes = [
    [(37.5, -120.5), (37.7, -120.7)],  # Route 1
    [(37.6, -120.6), (37.8, -120.8)]   # Route 2
]

# Draw evacuation routes
for route in evacuation_routes:
    folium.PolyLine(route, color="blue", weight=2.5, opacity=1).add_to(fire_map)

fire_map.save("fire_response_map.html")

