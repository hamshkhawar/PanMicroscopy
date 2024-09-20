#!/bin/bash

source /home/abbasih2/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate myvaex



singularity run \
        --bind /projects/PanMicroscopy:/projects/PanMicroscopy \
        /projects/PanMicroscopy/plugins/ome_converter_tool_0_3_3_dev1.sif \
        --inpDir=$1 \
        --filePattern=".*.png" \
        --fileExtension=.ome.tif \
        --outDir=$2

echo "Finshed converting images:$1"

python rxrx_nyxus.py --inpDir=$2 --outDir=$3

echo "Finshed extracting features: $1"




