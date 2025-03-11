#!/bin/bash

# Define paths
plates="/home/abbasih2/projects/PANMicroscopy_scripts/Idr/plates"
logs="/home/abbasih2/projects/PANMicroscopy_scripts/Idr/logs"
out="/projects/PanMicroscopy/data/Idr/omeconverted/"
studyname="idr0002-heriche-condensation"

# Create logs directory
study_logs="${logs}/${studyname}"
mkdir -p "$study_logs"

platescsv="${plates}/${studyname}.csv"
plates_list=()

# Read CSV and skip header
while IFS=, read -r PlateID Path; do
    PlateID=$(echo "$PlateID" | tr -d '\r' | xargs)
    Path=$(echo "$Path" | tr -d '\r' | xargs)

    # Skip header line
    if [[ "$PlateID" == "PlateID" ]]; then
        continue
    fi
    if [[ -d "$Path" ]]; then
        plates_list+=("$Path")
    else
        echo "Warning: Directory $Path does not exist. Skipping." >&2
    fi
done < <(tail -n +2 "$platescsv")

total_plates=${#plates_list[@]}
max_jobs=10  
job_count=0
job_pids=() 

# Process each plate
for plate in "${plates_list[@]}"; do
    echo "Processing plate: $plate"

    # Log files
    log_out="${study_logs}/$(basename "$plate").err"
    log_err="${study_logs}/$(basename "$plate").err"

    # Collect all .tif files in the plate
    mapfile -t tif_files < <(find -L "$plate" -type f -name "*.tif")

    # Divide files into groups of 2
    total_files=${#tif_files[@]}
    batch_size=100

    for ((i = 0; i < total_files; i += batch_size)); do
        file_batch=("${tif_files[@]:i:batch_size}")

        # Convert array to comma-separated string
        input_files=$(printf ",%s" "${file_batch[@]}")
        input_files=${input_files:1}  
        echo $input_files

        filepattern=".*W{w:d+}--P{p:d+}--Z{z:d+}--T{t:d+}--(?P<channel>.*).tif"
        
        echo "Submitting job for batch: $input_files"

       
        # srun -N 1 -p preempt_cpu --mem-per-cpu=20G --output="$log_out" --error="$log_err" --time=2-00:00:00 singularity run \
        srun -N 1 -p preempt_cpu --mem-per-cpu=20G --output="$log_out" --error="$log_err" singularity run \
            --env POLUS_IMG_EXT=".ome.zarr" \
            --bind /projects:/projects/ \
            /home/abbasih2/projects/plugins/polusai_ome-converter-tool:0.3.3-dev5.sif \
            --inpDir="$input_files" \
            --filePattern="$filepattern" \
            --outDir="$out" &

        # Store the PID of the last submitted job
        job_pids+=($!)
        job_count=$((job_count + 1))

        # If we've reached the max number of jobs, wait for all to complete
        if [[ $job_count -ge $max_jobs ]]; then
            echo "Reached max jobs ($max_jobs). Waiting for current jobs to finish..."
            wait "${job_pids[@]}"  # Wait for all jobs in the current batch to complete
            job_count=0  # Reset job count
            job_pids=()  # Clear the PID array for the next batch
            echo "Batch completed. Proceeding to next set of jobs..."
        fi
    done
done

# Wait for any remaining jobs to finish
if [[ ${#job_pids[@]} -gt 0 ]]; then
    echo "Waiting for remaining jobs to complete..."
    wait "${job_pids[@]}"
fi

echo "All jobs completed."
