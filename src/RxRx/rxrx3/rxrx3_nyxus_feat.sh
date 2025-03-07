#!/bin/bash
#SBATCH --job-name=rxrx3_feat              # Job name
#SBATCH -p extended_cpu                    # Partition name
#SBATCH --nodes=20                         # Number of nodes
#SBATCH --ntasks=20                        # Number of tasks (one per node)
#SBATCH --cpus-per-task=1                  # Number of CPUs per task
#SBATCH --mem=20G                          # Memory per node
#SBATCH --output=rxrx3_feat.txt            # Output file
#SBATCH --error=rxrx3_feat.txt             # Error log file
#SBATCH --time=14-00:00:00                 # Maximum runtime

source /home/abbasih2/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate myvaex

# Display the list of allocated nodes
echo "Nodes allocated to the job: $SLURM_JOB_NODELIST"
scontrol show hostname $SLURM_JOB_NODELIST > nodelist.txt

# Check if nodelist.txt is empty
if [[ ! -s nodelist.txt ]]; then
    echo "Error: nodelist.txt is empty. No nodes allocated or scontrol command failed."
    exit 1
fi

# Read nodes into an array
mapfile -t nodes < nodelist.txt
if [[ ${#nodes[@]} -eq 0 ]]; then
    echo "Error: No nodes found. The nodes array is empty."
    exit 1
fi

# Directory to search
search_dir="/projects/PanMicroscopy/data/recursionpharma/omeconverted/rxrx3"
nyxus_dir="/projects/PanMicroscopy/NyxusFeatures/recursionpharma/rxrx3"

# Find all Plate directories
find "$search_dir" -type d -name "*Plate*" > omedir.txt
mapfile -t my_list < omedir.txt
len=${#my_list[@]}

# Distribute jobs across nodes and process 20 files in parallel
batch_size=20
for ((i=0; i<len; i+=batch_size)); do
    for ((j=i; j<i+batch_size && j<len; j++)); do
        file=${my_list[$j]}
        part="${file#*rxrx3/}" 
        out_path="$nyxus_dir/$part"
        mkdir -p "$out_path"

        node_index=$((j % ${#nodes[@]}))
        
        echo "Submitting job for '$file' on node ${nodes[$node_index]}..."
        
        # Run feature extraction in parallel
        srun --nodes=1 --ntasks=1 --exclusive --nodelist=${nodes[$node_index]} \
        python /home/abbasih2/projects/PANMicroscopy_scripts/RxRx/rxrx3/rxrx3_nyxus.py --inpDir=${file} --outDir=${out_path} &
        
        # Print a message after processing each file
        echo "Feature extraction started for ${file}."
    done
    
    # Wait for the current batch of 20 jobs to complete before starting the next
    wait
    
    echo "Batch of 20 jobs completed."
done

echo "All feature extraction jobs completed."