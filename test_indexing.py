import numpy as np

lat = np.load('data/processed/grid_lat.npy')
lon = np.load('data/processed/grid_lon.npy')
grid = np.load('data/processed/forecast_grid_7day.npy')

test_lat = 12.9716
test_lon = 77.5946

print("lat shape:", lat.shape)
try:
    lat_idx = int(np.abs(lat - test_lat).argmin())
    print("Flat lat_idx:", lat_idx)
    
    # Let's see what unravel_index does
    idx_2d = np.unravel_index(np.abs(lat - test_lat).argmin(), lat.shape)
    print("Proper 2D index:", idx_2d)

    # BUT wait, the current code in main.py does:
    # lat_idx = int(np.abs(lat_axis - lat).argmin())
    # lon_idx = int(np.abs(lon_axis - lon).argmin())
    # hourly_aqi = forecast_grid[:, lat_idx, lon_idx]
    
    # If the user's code just uses argmin on a 2D array, let's see what value it gets
    print("Code lat_idx:", lat_idx)
    lon_idx = int(np.abs(lon - test_lon).argmin())
    print("Code lon_idx:", lon_idx)

    # Now if lat_idx is huge, wait, forecast_grid is shape (168, 241, 248)
    # Does it crash?
    try:
        val = grid[:, lat_idx, lon_idx]
        print("Success, val shape:", val.shape)
    except Exception as e:
        print("Crash indexing:", e)
        
except Exception as e:
    print(e)
