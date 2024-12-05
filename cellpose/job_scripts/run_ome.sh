#!/bin/bash


inp_dir="/projects/PanMicroscopy/data/cellpose/raw/train_cyto2"
out_path="/projects/PanMicroscopy/data/cellpose/omeconverted/train_cyto2"
file_pattern=".*_img.png"
    
   
 srun -N 1 -n 1 --mem-per-cpu=20G -p quick_cpu  singularity run \
                --env POLUS_IMG_EXT=".ome.tif" \
                --bind /projects/:/projects/ \
                /home/abbasih2/projects/plugins/ome-converter-tool_0.3.3-dev3.sif \
                --inpDir="$inp_dir" \
                --filePattern="$file_pattern" \
                --outDir="$out_path" &

