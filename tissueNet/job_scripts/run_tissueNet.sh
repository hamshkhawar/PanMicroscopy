#!/bin/sh

platespath="/home/abbasih2/projects/PANMicroscopy_scripts/tissueNet/plates.txt"
logs="/home/abbasih2/projects/PANMicroscopy_scripts/tissueNet/logs"

mkdir -p "$logs"

file_pattern=".*.tif"

# Initialize an empty array
plates_list=()

# Read each line from the text file into the array
while IFS= read -r line; do
    plates_list+=("$line")  # Append each line to the array
done < "$platespath"

# Loop through each plate and submit the job
for plate in "${plates_list[@]}"; do
    echo "Processing plate: $plate"

    # Extract the last segments of the plate path
    part=$(echo "$plate" | awk -F'/' '{print $(NF-3)"/"$(NF-2)"/"$(NF-1)"/"$NF}')
    
    # Create a unique log name by replacing slashes with underscores
    name="${part//\//_}"
    
    # Define paths for output and error logs
    out_log="$logs/${name}.out"
    err_log="$logs/${name}.err"
    
    echo "Submitting job for plate: $name"
    
    # Submit the job using srun
    srun -N 1 --gpus=1 --mem=20G -p extended_gpu -o "$out_log" -e "$err_log" --time=2-00:00:00 ./nyxus_gpu.sh "$plate" &
done

# Wait for all jobs to complete
wait

echo "All jobs completed."

