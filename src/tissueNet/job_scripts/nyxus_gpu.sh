#!/bin/sh

source /home/abbasih2/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate nyxusgpu

python /home/abbasih2/projects/PANMicroscopy_scripts/tissueNet/nyxus_gpu.py --inpDir=$1
