import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from netCDF4 import Dataset
import numpy as np
from datetime import datetime
import os 
from tqdm import tqdm

def plot_hm(x, y, z, fn, figsize=(8, 5)):
    # Create a vertical plot
    fig,ax = plt.subplots(figsize=figsize)
    # note: O3 in partial pressure
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
    plt.tight_layout()
    fig.savefig("./plot/" + fn, dpi=300)
    return

def plotLn(p,dt,dt_all,fn,error=None):
    def partialPtoPPM(dt_pp,air_p):
        if len(dt_pp)==len(air_p):
            return dt_pp / air_p *10
        else:
            print("Different length for dt and air p.")
        return None
    # Define the function to convert ozone mass density to ppmv
    def convert_ozone_mass_density_to_ppmv(mass_density_ozone, temperature, pressure):
        # Constants
        molar_mass_ozone = 48.00  # g/mol
        universal_gas_constant = const.R  # J/(mol*K)

        # Convert mass density to molar concentration (mol/m^3)
        molar_concentration_ozone = mass_density_ozone / molar_mass_ozone

        # Calculate the total number density of air (mol/m^3) using the ideal gas law
        number_density_air = pressure / (universal_gas_constant * temperature)

        # Calculate the volume mixing ratio (ppmv)
        vmr_ppmv = (molar_concentration_ozone / number_density_air) * 10**6

        return vmr_ppmv

    fig = plt.figure()
    for i,ln in enumerate(dt_all):
        #new_ln = partialPtoPPM(dt_pp=ln, air_p=p)
        new_ln = convert_ozone_mass_density_to_ppmv(mass_density_ozone=ln, temperature=288, pressure=p)
        #plt.plot(ln,p, linestyle='-',c='black',alpha=0.2,lw=5)
        plt.plot(new_ln,p, linestyle='-',c='black',alpha=0.2,lw=5)
    #plt.errorbar(dt,p,xerr=error,capsize=5,ecolor='red',c='red')
    plt.yscale('log')
    plt.gca().invert_yaxis()
    plt.xlabel('O3 [?]')
    plt.ylabel('Pressure (p-level.dat) [hPa]')
    #fig.savefig("./plot/" + 'O3_JRA55.jpg',dpi=300)
    plt.tight_layout()
    fig.savefig("./plot/" + fn,dpi=300)

def handle_pressure_nan(p):
    # find max length 
    p = np.array(p)
    for i in range(0,len(p)):
        if (~np.isnan(p[i]).any()):
            reference = p[i]
            #print(reference)
    # then, if nan found, assign reference p
    for i in range(0,len(p)):
        if (np.isnan(p[i]).any()):
            ind = np.where(np.isnan(p[i]))[0]
            p[i][ind] = reference[ind]
    return p

def remove_repeated_dates(org_dates):
    new_dates = set(org_dates)
    new_dates_list = list(new_dates)
    new_dates_list.sort()
    output = [date.replace('-', '/') for date in new_dates_list]
    return output

def plot_all_nc_files(nc_files: list, path="../data/"):
    for i,file in tqdm(enumerate(nc_files)):
        #file_path = 'Tsukuba-sonde.nc'
        file_path = path + file
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

        # Create a heatmap plot
        print("--------------------")
        plot_hm(x=dates, y=pressure_data.T, z=temperature_data.T,
                fn=file+'_temperature.jpg')
        print(f'{file}: Plotted temperature.jpg')

        # Create a heatmap plot
        plot_hm(x=dates, y=pressure_data.T, z=humidity_data.T,
                fn=file+'_humidity.jpg')
        print(f'{file}: Plotted humidity.jpg')

        # Create a heatmap plot
        plot_hm(x=dates, y=pressure_data.T, z=o3_data.T,
                fn=file+'_o3.jpg')
        print(f'{file}: Plotted o3.jpg')
        print("--------------------")

        # Close the NetCDF4 file
        dataset.close()

def main():
    # Open the NetCDF4 file
    path =  "../data/"
    nc_files = [f for f in os.listdir(path) if f.endswith('.nc')]
    # Plot all nc data
    plot_all_nc_files(nc_files=nc_files, path=path)
    return


if __name__ == "__main__":
    main()
# end