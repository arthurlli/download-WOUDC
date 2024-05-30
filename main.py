from pywoudc import WoudcClient
import pandas as pd
import json
from io import StringIO
from collections import defaultdict
import xarray as xr
from netCDF4 import Dataset
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def get_WOUDC_data(bbox: list, temporal: list, dt_type='ozonesonde'):
    """
    Download WOUDC ozonesonde data via API and store to geoJason.

    Args:
        bbox (list): a list of boundary box for get_data method in WoudcClient (minx, miny, maxx, maxy)
        temporal (list): a range of date in %y/%m%d format

    Returns:
        data_by_date (GeoJason): filtered ozonesonde data separated by date
    """
    # define woudc client
    client = WoudcClient()

    # download data
    dt = client.get_data(typename=dt_type,
                        #bbox=[139, 35.6, 141, 36.6],  # bbox of Tokyo
                        bbox=bbox,
                        temporal=temporal)

    features = dt['features']
    
    # Initialize a dictionary to hold temperature, ozone, pressure, and humidity data by date
    data_by_date = defaultdict(lambda: {'Pressure': [],
                                        'Temperature': [], 
                                        'RelativeHumidity': [],
                                        'O3PartialPressure': []})

    # Process each feature
    for feature in features:
        properties = feature['properties']
        instance_datetime = properties['instance_datetime']
        
        # Extract date part from the datetime string
        date = instance_datetime.split(' ')[0]
        
        # Read data_block into a DataFrame
        data_block = properties['data_block']
        df = pd.read_csv(StringIO(data_block))
        
        # Append temperature, ozone, pressure, and humidity data to the corresponding date
        data_by_date[date]['Pressure'].extend(df['Pressure'].tolist())
        data_by_date[date]['Temperature'].extend(df['Temperature'].tolist())
        data_by_date[date]['RelativeHumidity'].extend(df['RelativeHumidity'].tolist())
        data_by_date[date]['O3PartialPressure'].extend(df['O3PartialPressure'].tolist())
    return data_by_date

def print_data(data_by_date):
    # Print the separated data by date
    for date, data in data_by_date.items():
        print(f"Date: {date}")
        print("Pressures:", data['Pressure'])
        print("Temperatures:", data['Temperature'])
        print("Humidities:", data['RelativeHumidity'])
        print("Ozones:", data['O3PartialPressure'])
        print(f"Len(T): {len(data['Temperature'])}")
        print(f"Len(H): {len(data['RelativeHumidity'])}")
        print(f"Len(O3): {len(data['O3PartialPressure'])}")
        print(f"Len(P): {len(data['Pressure'])}")
    return

def save_to_NetCDF4(data_by_date, savename: str):
    """
    Store the retrieved data to NetCDF4 format, in the same struture as MIPAS O3.

    Args:
        data_by_date (GeoJason): WOUDC ozoneosnde data
        savename (str): filename of the .nc file
    """
    # Get the maximum length of data for each variable
    max_length = max(len(data) for date_data in data_by_date.values() for data in date_data.values())

    # Pad the data to the maximum length with NaN values
    for date_data in data_by_date.values():
        for variable, data in date_data.items():
            date_data[variable] = np.pad(data, (0, max_length - len(data)), mode='constant', constant_values=np.nan)

    # Convert data_by_date to xarray dataset
    dates = list(data_by_date.keys())
    variables = list(data_by_date[dates[0]].keys())

    # Create empty dataset
    ds = xr.Dataset()

    # Add variables to the dataset
    for variable in variables:
        data_array = np.vstack([data_by_date[date][variable] for date in dates])
        ds[variable] = xr.DataArray(data_array, dims=('date', 'target'), coords={'date': dates})

    # set units, std name, and long name
    units = {'Pressure': 'hPa',
            'Temperature': 'deg. C', 
            'RelativeHumidity': '%',
            'O3PartialPressure': 'ppb'}
    standard_name = {'Pressure': 'P',
                    'Temperature': 'T', 
                    'RelativeHumidity': 'RH',
                    'O3PartialPressure': 'O3'}
    long_name = {'Pressure': 'Pressure',
                'Temperature': 'Temperature', 
                'RelativeHumidity': 'RelativeHumidity',
                'O3PartialPressure': 'O3PartialPressure'}
    
    
    # Set attributes for variables
    for variable in variables:
        ds[variable].attrs = {
            'units': units[variable],
            'standard_name': standard_name[variable],
            'long_name': long_name[variable],
            'missing_value': np.nan
        }

    # Save the dataset to a NetCDF4 file
    output_filename = savename
    ds.to_netcdf(output_filename, format='NETCDF4_CLASSIC')
    # TODO testing
    #ds.to_csv(output_filename+'.csv')

    # Close the dataset
    ds.close()

    print(f"NetCDF4 file saved as: {output_filename}")
    return

def read_nc4(file_path: str):
    """
    A function to read nc4

    Args:
        file_path (str): path for the file

    Returns:
        dateset: the read xarray data
    """
    dataset = xr.open_dataset(file_path)
    return dataset

def save_dates_to_txt(file_path: str, attr='date'):
    """
    Store the dates values in .nc4 file to a txt file

    Args:
        file_path (str): path of .nc file
    """
    # Open the NetCDF file
    #file_path = 'Tsukuba-sonde.nc'
    dataset = read_nc4(file_path)

    # Extract the dates
    dates = dataset[attr].values

    # Convert dates to a list of strings
    dates_list = [str(date) for date in dates]
    #print(dates_list)

    # Save the dates to a text file
    output_txt_file = 'dates.txt'
    with open(output_txt_file, 'w') as f:
        for date in dates_list:
            f.write(f"{date}\n")

    print(f"Dates have been saved to {output_txt_file}")

    dataset.close()
    return

def plot_nc4(file_path: str):
    """
    Plot heat maps of the .nc4 file and check 2020/01/06 vertical profile

    Args:
        file_path (str): path of .nc file
    """

    def plot_hm(x, y, z, fn):
        # Create a vertical plot
        fig,ax = plt.subplots(figsize=(8, 5))
        c = ax.pcolormesh(x,y,z)
        # Add a color bar
        colorbar = fig.colorbar(c, ax=ax)
        plt.yscale('log')
        plt.gca().invert_yaxis()
        plt.ylim((np.max(pressure_data),0.1))
        # Rotate x-axis tick labels
        plt.xticks(rotation=45)
        # only integer ticks are shown on the x-axis
        ax.xaxis.set_major_locator(MaxNLocator(nbins=25))
        plt.xlabel('Dates')
        plt.ylabel('Pressure [hPa]')
        fig.savefig(fn, dpi=300)
        return

    def handle_pressure_nan(p):
        # find max length 
        p = np.array(p)
        for i in range(0,len(p)):
            if (~np.isnan(p[i]).any()):
                reference = p[i]
                print(reference)
        # then, if nan found, assign reference p
        for i in range(0,len(p)):
            if (np.isnan(p[i]).any()):
                ind = np.where(np.isnan(p[i]))[0]
                p[i][ind] = reference[ind]
        return p

    # Open the NetCDF4 file
    file_path = 'Tsukuba-sonde.nc'
    dataset = Dataset(file_path, 'r')

    # Extract time, pressure, and temperature data
    time_data = np.array(dataset.variables['date'][:])
    pressure_data =  np.array(dataset.variables['Pressure'][:])
    temperature_data =  np.array(dataset.variables['Temperature'][:])
    humidity_data =  np.array(dataset.variables['RelativeHumidity'][:])
    o3_data =  np.array(dataset.variables['O3PartialPressure'][:])

    # convert dates to datetime obj
    dates = np.array([datetime.strptime(str(date_str), "%Y/%m/%d") for date_str in time_data])

    # filter
    # Filter out non-finite or masked values
    valid_indices = np.isfinite(pressure_data) & np.isfinite(temperature_data)

    # handle nan in p: filled by max lengthed pressure
    pressure_data = handle_pressure_nan(p=pressure_data)

    # Create a vertical plot
    # TODO use the valid_indices to filter
    plot_hm(x=dates, y=pressure_data.T, z=temperature_data.T,
            fn='temperature.jpg')
    print('Plotted temperature.jpg')

    # Create a vertical plot
    plot_hm(x=dates, y=pressure_data.T, z=humidity_data.T,
            fn='humidity.jpg')
    print('Plotted humidity.jpg')

    # Create a vertical plot
    plot_hm(x=dates, y=pressure_data.T, z=o3_data.T,
            fn='o3.jpg')
    print('Plotted o3.jpg')


    # plot 2020/01/06
    ind = np.where(dates==datetime.strptime('2020/01/06', "%Y/%m/%d"))
    #print(ind)
    # Create a vertical plot
    fig,ax = plt.subplots(figsize=(8, 5))
    #print(o3_data[ind,:].ravel(),pressure_data[ind,:].ravel())
    #print("partial to ppm",o3_data[ind,:].ravel() / pressure_data[ind,:].ravel() *10)
    plt.plot(o3_data[ind,:].ravel() / pressure_data[ind,:].ravel() *10,pressure_data[ind,:].ravel())
    plt.yscale('log')
    plt.gca().invert_yaxis()
    #plt.ylim((np.max(pressure_data),0.1))
    fig.savefig('o3-20200106.jpg')

    # plot 2019/07/02
    ind = np.where(dates==datetime.strptime('2019/07/02', "%Y/%m/%d"))
    print(ind)
    # Create a vertical plot
    fig,ax = plt.subplots(figsize=(8, 5))
    print(o3_data[ind,:].ravel(),pressure_data[ind,:].ravel())
    print("partial to ppm",o3_data[ind,:].ravel() / pressure_data[ind,:].ravel() *10)
    plt.plot(o3_data[ind,:].ravel() / pressure_data[ind,:].ravel() *10,pressure_data[ind,:].ravel())
    plt.yscale('log')
    plt.gca().invert_yaxis()
    #plt.ylim((np.max(pressure_data),0.1))
    fig.savefig('o3-20190702.jpg')

    # Close the NetCDF4 file
    dataset.close()
    return

def main():
    # start [main]
    filename = 'Tsukuba-sonde.nc'
    data_by_date = get_WOUDC_data(bbox=[139, 35.6, 141, 36.6], # Tsukuba site
                                  temporal=['2019-01-01', '2023-12-31'])  # total 5 years data in stock
    save_to_NetCDF4(data_by_date=data_by_date, savename=filename)
    
    plot_nc4(file_path=filename)
    save_dates_to_txt(file_path=filename)
    # end [main]
    return


if __name__=="__main__":
    main()
# end