# TissueNet Data Parser

This project is a Python script designed to parse TissueNet dataset `.npz` files into intensity and label images, converting them into `.ome.tif` or .ome.zarr formats. It supports TissueNet versions `v1.0` and `v1.1`, organizes the output into a standardized directory structure, and leverages multiprocessing for efficient processing.


## Features

* Parses `.npz` files from the TissueNet dataset into intensity and label images.
* Supports TissueNet versions `v1.0` and `v1.1`.
* Outputs images in `.ome.tif` (default) or `.ome.zarr` formats using `bfio`.
* Organizes data into `raw` and `standard` subdirectories
* Uses multiprocessing to parallelize processing of multiple `.npz` files
* Includes optional normalization of image intensities.

## Prerequisites
- Python 3.9 or higher

## Installation

1. ### Clone the Repository
```bash
git clone <repository-url>
```

2. ### Install Dependencies

```bash
pip install bfio numpy tqdm
```
cd to working directory
```bash
cd  src/tissueNet/python_scripts
```

3.Set the `POLUS_EXT` environment variable (optional) to override the default file extension `(.ome.tif)`:

```bash
export POLUS_EXT=".ome.zarr"
```
## Arguments
* `root_directory (str)`: Path to the root directory containing raw and standard subdirectories
* `version (str)`: TissueNet version (v1.0 or v1.1)
* `file_extension (str)`: Output file format (e.g., .ome.tif or .ome.zarr).
* `normalize (bool)`: Whether to normalize image intensities (True or False)

## Usage

Run the script from the command line with the required arguments:
```bash
python tissue_standard.py <root_directory> <version> <file_extension> <normalize>
```

