# woudc-ozonesonde-downloader
A simple program to extract global ozonesonde data using API service provided by WOUDC.

# Repository structure
```
woudc-data-downloader/
├── src/             # Source code
│   └── main.py      # Main script
├── data/            # Data storage (for netCDF4 output)
├── README.md        # Project overview
├── requirements.txt # Dependencies
└── LICENSE          # MIT
```

# Overview
Collecting ozonesonde dataset is a crucial step for the validation of satellite retrieval and the investigation of long-term trend of atmospheric parameters.</br>
Yet, downloading ozonesonde data from WOUDC can be tidious. To simplify the process, this program aims to automatically download WOUDC ozonesonde data via offical python package of WOUDC (pywoudc) in netCDF4 format. </br>

# Pre-requisite
Installation of pywoudc package is required before running this script. Please visit https://github.com/woudc/pywoudc/tree/master for installation guide.

# NetCDF4 filename
The output file is named as [ID]_[location].nc, where [ID] is composed of latitude (9 digits + N/S sign) and longitude (9 digits + E/W sign).</br>
(For example, sonde data at Tsukuba, Japan is named "036049999N140133331E_Tateno(Tsukuba).nc".</br>

# Program workflow: overview
```
        +----------------------+
        |       Start           |
        |  (Import Modules)     |
        +----------------------+
                  |
                  v
        +-----------------------+
        | Extract WOUDC Metadata |
        +-----------------------+
                  |
                  v
        +-------------------------------+
        |     Retrieve Ozonesonde Data   |
        +-------------------------------+
                  |
                  v
  +--------------------------------------------+
  |  3.1 Define Bounding Box (+- 0.1 deg.)     |
  +--------------------------------------------+
                  |
                  v
  +--------------------------------------------+
  |  3.2 Organize Data in Multidimensional     |
  +--------------------------------------------+
                  |
                  v
  +--------------------------------------------+
  |  3.3 Clean Data (Remove Symbols)           |
  +--------------------------------------------+
                  |
                  v
  +--------------------------------------------+
  |  3.4 Generate Data ID (Lat/Long)           |
  +--------------------------------------------+
                  |
                  v
        +----------------------+
        |   Store Data (NetCDF4)|
        +----------------------+
                  |
                  v
        +----------------------+
        |        End            |
        +----------------------+
```

# Example
After extracting all sonde data in 2019 - 2023.</br>
option 1: If you want to check the contents of all NetCDF4 data </br>
```
cd src
python check_nc.py
cd ../
```
A new .txt file will be saved at ```./src/``` folder. </br>
The summary looks like: </br>
```
Dataset Information:
--------------------
File Path: 036049999N140133331E_Tateno(Tsukuba).nc
Available data period: 2019/01/04 - 2023/12/27, 206 days
Dimensions: ['date', 'target']
Variables:
  date:
    Shape: (206,)
    Data Type: object
    Attributes: []
  Pressure:
    Shape: (206, 7032)
    Data Type: float64
    Attributes: ['units', 'standard_name', 'long_name']
  Temperature:
    Shape: (206, 7032)
    Data Type: float64
    Attributes: ['units', 'standard_name', 'long_name']
  RelativeHumidity:
    Shape: (206, 7032)
    Data Type: float64
    Attributes: ['units', 'standard_name', 'long_name']
  O3PartialPressure:
    Shape: (206, 7032)
    Data Type: float64
    Attributes: ['units', 'standard_name', 'long_name']
--------------------
Statistics for file: 036049999N140133331E_Tateno(Tsukuba).nc
            Pressure    Temperature       Humidity          Ozone
count  706132.000000  706132.000000  207995.000000  706132.000000
mean      235.910875     -41.083497      34.019885       6.603494
std       268.572703      22.482677      27.710239       4.383417
min         3.600000     -77.200000       0.000000       0.360000
25%        29.400000     -57.300000       9.000000       2.940000
50%       112.700000     -48.800000      27.000000       5.010000
75%       368.500000     -30.700000      56.000000      10.250000
max      1027.400000      34.100000     100.000000      23.310000
```

option 2: if you want to check the geopgraphical location of each data
```
cd src
python plot_location.py 
cd ../
```
A new plot will be saved at ```./plot/``` as below: </br>
![WOUDC Sonde Map (2019-2023)](WOUDC_sonde_map.jpg)

option 3: if you want to visualise all the .nc data </br>
```
cd src
python plot_var.py 
cd ../
```
A series of plots will be saved at ```./plot/```. </br>
For example, 3 figures will be generated for ```036049999N140133331E_Tateno(Tsukuba).nc``` (see below). </br>
![temperature at Tsukuba (2019-2023)](036049999N140133331E_Tateno(Tsukuba).nc_temperature.jpg)
![humidity at Tsukuba (2019-2023)](036049999N140133331E_Tateno(Tsukuba).nc_humidity.jpg)
![ozone at Tsukuba (2019-2023)](036049999N140133331E_Tateno(Tsukuba).nc_o3.jpg)