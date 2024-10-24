import netCDF4 as nc
import numpy as np
import os 

# Load the NetCDF4 file
path = '../data/'
#filename = '025020000N121480003E_Taipei.nc'
nc_files = [f for f in os.listdir(path) if f.endswith('.nc') ]

for i, filename in enumerate(nc_files):
    # Load the NetCDF4 file
    dataset = nc.Dataset(path + filename)

    # Extract variables (assuming they are 2D: [time, level])
    pressure = dataset.variables['Pressure'][:]
    temperature = dataset.variables['Temperature'][:]
    relative_humidity = dataset.variables['RelativeHumidity'][:]
    o3_partial_pressure = dataset.variables['O3PartialPressure'][:]
    dates = dataset.variables['date']

    # List of variables to check
    variables = {
        'Pressure': pressure,
        'Temperature': temperature,
        'RelativeHumidity': relative_humidity,
        'O3PartialPressure': o3_partial_pressure
    }

    # Function to calculate modified z-score
    def modified_z_score(data):
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return np.zeros(data.shape)
        return 0.6745 * (data - median) / mad

    # Function to detect outliers using modified Z-score for each date
    def detect_outliers_mod_zscore(data, file, var_name, dates, threshold=3.5):
        # Set NumPy print options to avoid truncation
        np.set_printoptions(threshold=np.inf)

        for i in range(data.shape[0]):  # Loop over time/dates
            z_scores = modified_z_score(data[i, :])
            
            # Identify outliers for the current date
            outliers = np.where(np.abs(z_scores) > threshold)[0]

            if len(outliers) > 0:
                file.write("\n")
                file.write(f"Outliers in {var_name} for date {dates[i]} (threshold: {threshold}):\n")
                file.write(f"Outliers at levels: {outliers}\n")
                file.write(f"Outlier values: {data[i, outliers]}\n")
                file.write("\n")
            else:
                file.write(f"No outliers in {var_name} for date {dates[i]}.\n")


    # Write outliers to a file
    output_file = path + filename.replace('.nc','_') + 'outliers_check.txt'
    with open(output_file, 'w') as file:
        # Apply the Modified Z-Score method for each variable
        for var_name, var_data in variables.items():
            file.write(f"--- Checking outliers for {var_name} ---\n")
            detect_outliers_mod_zscore(var_data, file, var_name, dates)

    # Close the dataset
    dataset.close()

    print(f"Outlier information by date has been written to {output_file}")
