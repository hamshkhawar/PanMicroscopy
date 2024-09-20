#!/bin/sh

#SBATCH --job-name=untar            # Job name
#SBATCH -p extended_cpu              # Partition name
#SBATCH --output=untar.txt          # Standard output and error log
#SBATCH --error=untar.txt           # Error log file
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem-per-cpu=8G

source /home/abbasih2/miniconda3/etc/profile.d/conda.sh

conda init bash

conda activate myenv

# Directory to search
search_dir="/projects/PanMicroscopy/data/recursionpharma/raw/rxrx3/images"

# Find all .tar files and store them in a file
find "$search_dir" -type f -name "*.tar" > tar_files.txt

# Define a function to extract a single tar file
extract_file() {
    file=$1
    base_name=$(basename "$file" .tar)
    new_dir="$(dirname "$file")/$base_name"
    mkdir -p "$new_dir"
    echo "Extracting '$file' into '$new_dir'..."
    tar -xf "$file" -C "$new_dir"
    rm "$file"
    echo "Finished extracting and removed '$file'."
}

export -f extract_file

# Use xargs to parallelize extraction
cat tar_files.txt | xargs -n 1 -P $SLURM_CPUS_PER_TASK -I {} bash -c 'extract_file "$@"' _ {}

# Cleanup
rm tar_files.txt

