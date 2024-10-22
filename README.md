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

