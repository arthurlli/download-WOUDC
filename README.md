# download-WOUDC
A simple program to download WOUDC ozonesonde data over Tokyo using API.

woudc-data-downloader/
├── src/             # Source code
│   └── main.py      # Main script
├── data/            # Data storage (for netCDF4 output)
├── README.md        # Project overview
├── requirements.txt # Dependencies
└── LICENSE          # MIT

# Content
For validating GOSAT-2 retrieval results, comparison with sonde measurement is always an effective solution. </br>
Yet, downloading ozonesonde data from WOUDC can be tidious. </br>
To simplify the process, this program aims to automatically download WOUDC ozonesonde data via offical python package of WOUDC (pywoudc).</br>
Also, the data are stored in netCDF4 format. </br>
