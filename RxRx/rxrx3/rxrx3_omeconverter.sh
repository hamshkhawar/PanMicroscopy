#!/bin/sh
#SBATCH --job-name=ome # Job name
#SBATCH -p extended_cpu      # Partition name
#SBATCH --output=ome.txt  # Standard output and error log
#SBATCH --error=ome.txt    # Error log file
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1             # Number of tasks (processes)
#SBATCH --cpus-per-task=1           # Number of CPU cores per task

search_dir="/projects/PanMicroscopy/data/recursionpharma/raw/rxrx3/images"

find "$search_dir" -type d -name "Plate*" | sort > tar_dir.txt

# Store the list in a variable and loop through it
dir_list=$(cat tar_dir.txt)

for dir in $dir_list; do
    echo "Processing directory: $dir"
    out_path="${dir/raw/omeconverted}"
    out_path=$(echo "$out_path" | sed 's/images//g')
     # Remove any redundant slashes
    out_path=$(echo "$out_path" | sed 's/\/\{2,\}/\//g')
    mkdir -p $out_path
    echo "output directory created: $out_path"

    singularity run \
            --bind /projects/PanMicroscopy:/projects/PanMicroscopy \
            /projects/PanMicroscopy/plugins/ome-converter-tool_0_3_2_dev1.sif \
            --inpDir=${dir} \
            --filePattern=".*.png" \
            --fileExtension=.ome.tif \
            --outDir=${out_path}


done
