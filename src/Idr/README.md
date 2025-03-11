# IDR Metadata Fetcher

This IDR metadata model fetches plate metadata (TSV files) and annotation data (CSV files, including gzipped ones) from the [IDR/idr-metadata](https://github.com/IDR/idr-metadata/tree/master) GitHub repository. It processes submodules and main repository contents, saving the data as CSV files in a specified output directory


## Features

* Retrieves `.gitmodules` content to identify submodules
* Fetches TSV files (`filePaths.tsv`, `plates.tsv`) and converts them to CSV with modified paths.
* Downloads and processes CSV files (`annotation.csv`).

## Prerequisites
- Python 3.9 or higher
- A GitHub personal access token (stored as an environment variable ACCESS_TOKEN)
- Poetry for dependency management


## Installation

1. ### Clone the Repository
```bash
git clone <repository-url>
```

2. ### Install Dependencies with Poetry
Ensure Poetry is installed [(installation guide)](https://python-poetry.org/docs/#installing-with-pipx), then run:
```bash
poetry install
```
cd to working directory
```bash
cd  src/Idr
```

3. ### Set Up Environment Variables
Create a `.venv` file in the project root or set the environment variable directly
```bash
echo "ACCESS_TOKEN=your_github_token" > .venv
```
## Usage

The script is run via a command-line interface using Typer. It requires three arguments:

`--root`: The root directory for path modifications. \
`--name`: The name of the IDR dataset (submodule or directory in the repository). \
`--outDir`: The output directory where processed files will be saved.


```bash
python idr_metadata_model.py --root <root-path> --name <dataset-name> --outDir <output-path>
```
## What It Does

-- Connects to the `IDR/idr-metadata` GitHub repository using the provided `ACCESS_TOKEN` \
-- Checks for a `.gitmodules` file to locate the specified dataset as a submodule \
-- If found as a submodule 
- Retrieves the submodule’s commit hash. 
- Downloads TSV and CSV files from the submodule at that commit. 

-- If not a submodule:
- Searches the main repository for the dataset directory and processes its files.
- Saves processed files:
- TSV files as CSV in <outDir>/plates/.
- CSV files (processed from raw or gzipped content) in <outDir>/annotations/


## Output Structure

```text
/path/to/output/
├── plates/
│   ├── filePaths.csv
│   ├── plates.csv
│   └── ...
└── annotations/
    ├── annotation.csv
    ├── annotations.csv
    └── ...
```

## Path Validator and File Counter
This Python script processes a CSV file containing plate IDs and file paths, validates the existence of each path, counts files recursively, and updates paths if they’re not found.


## Overview

* Reads a `CSV` file with PlateID and Path columns
* Checks if each path exists.
* Counts files recursively in each directory.
* Searches recursively for updated paths if the original doesn’t exist.
* Outputs `CSV` file with Updated paths, subdirectories with file counts


```bash
python idr_validation_model.py --inFile /path/to/plates.csv
```
