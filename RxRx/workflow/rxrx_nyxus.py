import time
from pathlib import Path
import logging
import numpy as np
import typer
from nyxus import Nyxus
from tqdm import tqdm
import filepattern as fp
from bfio import BioReader
import vaex
import re
import os


app = typer.Typer()

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("Extract rxrx nyxus features")
logger.setLevel(logging.INFO)


#Note Gabor features are disable as they are computational extensive Moment features are calculated based on intenisty in old nyxus which is used here

varlist = [
        'COV', 'COVERED_IMAGE_INTENSITY_RANGE', 
        'ENERGY', 'ENTROPY', 'EXCESS_KURTOSIS', 'HYPERFLATNESS', 'HYPERSKEWNESS', 
        'INTEGRATED_INTENSITY', 'INTERQUARTILE_RANGE', 'KURTOSIS', 'MAX', 'MEAN', 
        'MEAN_ABSOLUTE_DEVIATION', 'MEDIAN', 'MEDIAN_ABSOLUTE_DEVIATION', 'MIN', 'MODE',
        'P01', 'P10', 'P25', 'P75', 'P90', 'P99', 'QCOD', 'RANGE', 'ROBUST_MEAN', 
        'ROBUST_MEAN_ABSOLUTE_DEVIATION', 'ROOT_MEAN_SQUARED', 'SKEWNESS', 'STANDARD_DEVIATION', 
        'STANDARD_DEVIATION_BIASED', 'STANDARD_ERROR', 'VARIANCE', 'VARIANCE_BIASED', 'UNIFORMITY',
        'UNIFORMITY_PIU', 'GLCM_ASM', 'GLCM_ACOR','GLCM_CLUPROM', 'GLCM_CLUSHADE', 'GLCM_CLUTEND', 'GLCM_CONTRAST', 
        'GLCM_CORRELATION', 'GLCM_DIFAVE', 'GLCM_DIFENTRO', 'GLCM_DIFVAR', 'GLCM_DIS','GLCM_ENERGY',
        'GLCM_ENTROPY', 'GLCM_HOM1', 'GLCM_HOM2', 'GLCM_ID', 'GLCM_IDN', 'GLCM_IDM', 'GLCM_IDMN', 'GLCM_INFOMEAS1', 
        'GLCM_INFOMEAS2','GLCM_IV', 'GLCM_JAVE', 'GLCM_JE', 'GLCM_JMAX', 'GLCM_JVAR', 
        'GLCM_SUMAVERAGE', 'GLCM_SUMENTROPY', 'GLCM_SUMVARIANCE', 'GLCM_VARIANCE', 'GLCM_ASM_AVE', 'GLCM_ACOR_AVE', 
        'GLCM_CLUPROM_AVE', 'GLCM_CLUSHADE_AVE', 'GLCM_CLUTEND_AVE', 'GLCM_CONTRAST_AVE', 'GLCM_CORRELATION_AVE', 'GLCM_DIFAVE_AVE', 'GLCM_DIFENTRO_AVE', 
        'GLCM_DIFVAR_AVE', 'GLCM_DIS_AVE', 'GLCM_ENERGY_AVE', 'GLCM_ENTROPY_AVE', 'GLCM_HOM1_AVE', 'GLCM_ID_AVE', 
        'GLCM_IDN_AVE', 'GLCM_IDM_AVE', 'GLCM_IDMN_AVE', 'GLCM_IV_AVE', 'GLCM_JAVE_AVE', 'GLCM_JE_AVE', 
        'GLCM_INFOMEAS1_AVE', 'GLCM_INFOMEAS2_AVE', 'GLCM_VARIANCE_AVE', 'GLCM_JMAX_AVE', 'GLCM_JVAR_AVE', 
        'GLCM_SUMAVERAGE_AVE', 'GLCM_SUMENTROPY_AVE', 'GLCM_SUMVARIANCE_AVE', 'FRAC_AT_D', 
        'MEAN_FRAC', 'RADIAL_CV', 'ZERNIKE2D', 'SPAT_MOMENT_00', 'SPAT_MOMENT_01', 'SPAT_MOMENT_02', 'SPAT_MOMENT_03',
        'SPAT_MOMENT_10', 'SPAT_MOMENT_11', 'SPAT_MOMENT_12', 'SPAT_MOMENT_13', 'SPAT_MOMENT_20', 'SPAT_MOMENT_21', 
        'SPAT_MOMENT_22', 'SPAT_MOMENT_23', 'SPAT_MOMENT_30', 'CENTRAL_MOMENT_00', 'CENTRAL_MOMENT_01', 'CENTRAL_MOMENT_02', 
        'CENTRAL_MOMENT_03', 'CENTRAL_MOMENT_10', 'CENTRAL_MOMENT_11', 'CENTRAL_MOMENT_12', 
        'CENTRAL_MOMENT_13', 'CENTRAL_MOMENT_20', 'CENTRAL_MOMENT_21', 'CENTRAL_MOMENT_22', 'CENTRAL_MOMENT_23', 
        'CENTRAL_MOMENT_30', 'CENTRAL_MOMENT_31', 'CENTRAL_MOMENT_32', 'CENTRAL_MOMENT_33', 'NORM_SPAT_MOMENT_00',
        'NORM_SPAT_MOMENT_01', 'NORM_SPAT_MOMENT_02', 'NORM_SPAT_MOMENT_03', 'NORM_SPAT_MOMENT_10', 'NORM_SPAT_MOMENT_11', 
        'NORM_SPAT_MOMENT_12', 'NORM_SPAT_MOMENT_13', 'NORM_SPAT_MOMENT_20', 'NORM_SPAT_MOMENT_21', 'NORM_SPAT_MOMENT_22', 
        'NORM_SPAT_MOMENT_23', 'NORM_SPAT_MOMENT_30', 'NORM_SPAT_MOMENT_31', 'NORM_SPAT_MOMENT_32', 'NORM_SPAT_MOMENT_33', 
        'NORM_CENTRAL_MOMENT_02', 'NORM_CENTRAL_MOMENT_03', 'NORM_CENTRAL_MOMENT_11', 'NORM_CENTRAL_MOMENT_12', 
        'NORM_CENTRAL_MOMENT_20', 'NORM_CENTRAL_MOMENT_21', 'NORM_CENTRAL_MOMENT_30', 'HU_M1', 'HU_M2', 'HU_M3', 
        'HU_M4', 'HU_M5', 'HU_M6', 'HU_M7', 'WEIGHTED_SPAT_MOMENT_00', 'WEIGHTED_SPAT_MOMENT_01', 'WEIGHTED_SPAT_MOMENT_02', 
        'WEIGHTED_SPAT_MOMENT_03', 'WEIGHTED_SPAT_MOMENT_10', 'WEIGHTED_SPAT_MOMENT_11', 'WEIGHTED_SPAT_MOMENT_12', 
        'WEIGHTED_SPAT_MOMENT_20', 'WEIGHTED_SPAT_MOMENT_21', 'WEIGHTED_SPAT_MOMENT_30', 'WEIGHTED_CENTRAL_MOMENT_02', 
        'WEIGHTED_CENTRAL_MOMENT_03', 'WEIGHTED_CENTRAL_MOMENT_11', 'WEIGHTED_CENTRAL_MOMENT_12', 'WEIGHTED_CENTRAL_MOMENT_20',
        'WEIGHTED_CENTRAL_MOMENT_21', 'WEIGHTED_CENTRAL_MOMENT_30', 'WEIGHTED_HU_M1', 'WEIGHTED_HU_M2', 'WEIGHTED_HU_M3',
        'WEIGHTED_HU_M4', 'WEIGHTED_HU_M5', 'WEIGHTED_HU_M6', 'WEIGHTED_HU_M7'
        # 'GABOR', 
        ]




def nyxfun(intensity_dir, file_pattern, out_dir, pixels_per_micron):

    nyx = Nyxus(varlist)

    nyx_params = {
        "neighbor_distance": 5,
        "pixels_per_micron": pixels_per_micron,
        "n_feature_calc_threads": 8,
    }

    nyx.set_params(**nyx_params)

    
    nyx.featurize_directory(intensity_dir=str(intensity_dir), 
                            label_dir=None,
                            file_pattern=file_pattern,
                            output_type = "arrowipc",
                            output_path = str(out_dir))
    

@app.command()
def main(
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        help="Input generic data collection to be processed by this plugin",
        exists=True,
        resolve_path=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        help="Output collection",
        exists=True,
        resolve_path=True,
        writable=True,
        file_okay=False,
        dir_okay=True,
    )
    ):

    starttime = time.time()

    fps = fp.FilePattern(inp_dir, ".*.ome.tif")
    image_path = [f[1][0] for f in fps()]
    if len(image_path) > 0:
        br = BioReader(image_path[0])
        if br.physical_size_y[0] is not None:
            pixels_per_micron = br.physical_size_y[0]
        else:
            pixels_per_micron = 1.0

    logger.info(f'pixels_per_micro: {pixels_per_micron}')
    dataset = inp_dir.parent.parent.name
    experiment = inp_dir.parent.name
    plate = re.findall(r'\d+',  inp_dir.name)[0]


    nyxfun(intensity_dir=inp_dir, file_pattern=".*.ome.tif", out_dir=out_dir, pixels_per_micron = pixels_per_micron)

    arrowpath = Path(out_dir, 'NyxusFeatures.arrow')

    x = vaex.open(arrowpath)
    x['dataset'] =  np.array([dataset] * len(x))
    x['experiment'] =  np.array([experiment] * len(x))
    x['plate'] =  np.array([plate] * len(x))
    x['path'] =  np.array([str(inp_dir)] * len(x))

    x.export_feather(out_dir.joinpath(f"NyxusFeatures.arrow"))

    finishtime = (time.time() - starttime) / 60
    logger.info(f'total time taken in minutes {finishtime}')
    


if __name__ == '__main__':
    app()


    

