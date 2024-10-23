import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from adjustText import adjust_text

# Function to extract lat and lon from the filename
def extract_lat_lon_name(filename):
    pattern = r'(?P<lat>\d{9}[NS])(?P<lon>\d{9}[EW])_(?P<name>.+)\.nc'
    match = re.search(pattern, filename)
    if match:
        lat = match.group('lat')
        lon = match.group('lon')
        location_name = match.group('name')

        # Convert to float considering hemisphere
        lat_val = float(lat[:-1]) / 1e6 * (-1 if lat[-1] == 'S' else 1)
        lon_val = float(lon[:-1]) / 1e6 * (-1 if lon[-1] == 'W' else 1)
        
        return lat_val, lon_val, location_name
    return None

def list_out_nc(directory="../data/"):
    # List all .nc files in the directory
    nc_files = [f for f in os.listdir(directory) if f.endswith('.nc')]
    return nc_files

files = list_out_nc(directory="../data/")
locations = [extract_lat_lon_name(f) for f in files if extract_lat_lon_name(f) is not None]

# Prepare the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create a Basemap instance
#m = Basemap(projection='robin', resolution='l',
#            llcrnrlat=-90, urcrnrlat=90,
#            llcrnrlon=-180, urcrnrlon=180, ax=ax)
m = Basemap(projection='robin',lon_0=0,resolution='c')
# Draw coastlines and countries
m.drawcoastlines()
m.drawcountries()
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,360.,60.))
# Plot the locations and add location names
texts = []
for lat, lon, name in locations:
    print(lat,lon,name)
    x, y = m(lon, lat)
    m.plot(x, y, 'bo', markersize=8)

    #text_obj = plt.text(x, y, name, fontsize=8, ha='right', va='bottom', color='red')
    text_obj = plt.text(x, y, name, fontsize=8, color='red')
    texts.append(text_obj)

# Adjust the text to avoid overlaps
adjust_text(texts, army=True, only_move={'points':'y', 'text':'y'}, 
            force_text=0.1, force_points=0.1, 
            expand_text=(1.05, 1.05), expand_points=(1.05, 1.05))


plt.title('WOUDC sonde (2019-2023)')
plt.show()
fig.tight_layout()
fig.savefig('../plot/WOUDC_sonde_map.jpg',dpi=300)