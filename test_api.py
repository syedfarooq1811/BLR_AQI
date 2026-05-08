import numpy as np

grid_lat = np.load('data/processed/grid_lat.npy')
grid_lon = np.load('data/processed/grid_lon.npy')
grid = np.load('data/processed/forecast_grid_7day.npy')

lat_axis = grid_lat[:, 0]
lon_axis = grid_lon[0, :]

def check_coord(lat, lon, name):
    lat_idx = int(np.abs(lat_axis - lat).argmin())
    lon_idx = int(np.abs(lon_axis - lon).argmin())
    val = grid[0, lat_idx, lon_idx]
    actual_lat = lat_axis[lat_idx]
    actual_lon = lon_axis[lon_idx]
    print(f"{name} -> idx: ({lat_idx}, {lon_idx}), snapped: ({actual_lat:.4f}, {actual_lon:.4f}), val: {val:.2f}")

# Test coordinates
# Top right
check_coord(13.08, 77.69, "Top Right")
# Bottom left
check_coord(12.88, 77.48, "Bottom Left")
# Middle
check_coord(12.9716, 77.5946, "Center")

# Wait, let's check if grid_lat[:, 0] actually varies!
print("lat_axis first 5:", lat_axis[:5])
print("lon_axis first 5:", lon_axis[:5])
