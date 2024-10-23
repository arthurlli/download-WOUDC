import xarray as xr
import os 
import pandas as pd
import numpy as np

# List all .nc files
path = "../data"
nc_files = [f for f in os.listdir(path) if f.endswith('.nc') ]

# Open a file to save the output
with open("all_nc_info.txt", "w") as f:  # Specify the output text file name
    # Open the NetCDF4 file
    for i, file in enumerate(nc_files):
        fn = file
        ds = xr.open_dataset(f"{path}/{fn}", engine='netcdf4')

        # Print dataset information and save to file
        f.write("Dataset Information:\n")
        f.write("--------------------\n")
        f.write(f"File Path: {fn}\n")
        f.write(f"Available data period: {np.array(ds.variables['date'])[0]} - {np.array(ds.variables['date'])[-1]}, {len(np.array(ds.variables['date']))} days\n")
        f.write(f"Dimensions: {list(ds.dims.keys())}\n")

        # Print variables and their attributes, and save to file
        f.write("Variables:\n")
        for var_name in ds.variables:
            var = ds.variables[var_name]
            f.write(f"  {var_name}:\n")
            f.write(f"    Shape: {var.shape}\n")
            f.write(f"    Data Type: {var.dtype}\n")
            f.write(f"    Attributes: {list(var.attrs.keys())}\n")
        f.write("--------------------\n")

        # Transform to pd.DataFrame
        pressure = np.array(ds.variables['Pressure'][:]).ravel()
        temperature = np.array(ds.variables['Temperature'][:]).ravel()
        humidity = np.array(ds.variables['RelativeHumidity'][:]).ravel()
        ozone = np.array(ds.variables['O3PartialPressure'][:]).ravel()

        df = pd.DataFrame({
                    'Pressure': pressure,
                    'Temperature': temperature,
                    'Humidity': humidity,
                    'Ozone': ozone
                })

        # Compute statistics and save to file
        f.write(f"Statistics for file: {fn}\n")
        f.write(df.describe().to_string())  # Convert DataFrame stats to string and write to file
        f.write("\n\n")

        # Close the dataset
        ds.close()
