#!/bin/sh

dataset="rxrx1"
data_dir="/projects/PanMicroscopy/data/recursionpharma/raw/$dataset/$dataset/images"
logs="/home/abbasih2/submit_scripts/$dataset_logs"
mkdir -p "$logs"
nyxdir="/projects/PanMicroscopy/NyxusFeatures/recursionpharma/$dataset"

# find "$data_dir" -type d -name "Plate*" | sort > "${dataset}_plates.txt"

# # # Create a list
dir_list=$(cat "${dataset}_plates.txt")

for dir in $dir_list; do
    echo "Processing directory: $dir"
    # Replace "raw" with "omeconverted", replace duplicate "rxrx1", and remove "images"
    out_path=$(echo "$dir" | sed 's/raw/omeconverted/' | sed "s/$dataset\/$dataset/$dataset/" | sed 's/\/images//')
    dirname=$(echo "$dir" | sed 's|.*images/||')
    name=$(echo "$dirname" | sed 's|/|_|g')

    #Create directories if donot exist
    if [ ! -d "$out_path" ]; then
        mkdir -p "$out_path"
        echo "Directory $out_path created."
    else
        echo "Directory $out_path already exists."
    fi
    # Create a nyxus directory
    nyx_dir=$nyxdir/$dirname


    # Check if the directory exists, if not, create it
    if [ ! -d "$nyx_dir" ]; then
        mkdir -p "$nyx_dir"
        echo "Directory $nyx_dir created."
    else
        echo "Directory $nyx_dir already exists."
    fi

    # Define paths for output and error logs
    out_log="${logs}/${name}.txt"
    err_log="${logs}/${name}.txt"

    # # Submit the job using srun
    srun -N 1 -n 1 --mem-per-cpu=20G -p normal_cpu --cpus-per-task=4 -o "$out_log" -e "$err_log"  ./rxrx_containers.sh "$dir" "$out_path" "$nyx_dir" &



done
