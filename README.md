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

# Content
Collecting ozonesonde dataset is a crucial step for the validation of satellite retrieval and the investigation of long-term trend of atmospheric parameters.</br>
Yet, downloading ozonesonde data from WOUDC can be tidious. </br>
To simplify the process, this program aims to automatically download WOUDC ozonesonde data via offical python package of WOUDC (pywoudc) in netCDF4 format. </br>
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
  |  3.1 Define Bounding Box                   |
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