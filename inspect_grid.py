import numpy as np

try:
    grid = np.load('data/processed/forecast_grid_7day.npy')
    lat = np.load('data/processed/grid_lat.npy')
    lon = np.load('data/processed/grid_lon.npy')
    
    print("Grid shape:", grid.shape)
    print("Lat shape:", lat.shape, "Min/Max:", lat.min(), lat.max())
    print("Lon shape:", lon.shape, "Min/Max:", lon.min(), lon.max())
    
    # Check if grid has variation
    print("Grid Min/Max:", np.nanmin(grid), np.nanmax(grid))
    print("Grid Mean:", np.nanmean(grid))
    print("Grid Std:", np.nanstd(grid))
    
    # Check variation across space for hour 0
    hour0 = grid[0, :, :]
    print("Hour 0 Min/Max:", np.nanmin(hour0), np.nanmax(hour0))
    print("Hour 0 Std:", np.nanstd(hour0))

except Exception as e:
    print(e)
