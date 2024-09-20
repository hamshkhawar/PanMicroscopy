from pathlib import Path
import numpy as np
import vaex
import re
import glob
import os
from tqdm import tqdm
import logging
import gc
import time



logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("Combine rxrx2 features")
logger.setLevel(logging.INFO)



# Channel Dictionary
channel_dict = {
    '1': 'Hoechst-33342',
    '2': 'Concanavalin A', 
    '3': 'Phalloidin', 
    '4': 'Syto14', 
    '5': 'MitoTracker', 
    '6': 'WGA'
}

# Metadata columns
metacols = [
    'path',
    'dataset',
    'experiment',
    'plate',
    'wellnumber',
    'well',
    'site',
    'z_position',
    'channelnumber',
    'channel',
    'cell',
    'control_type',
    'perturbation',
    'perturbation_id',
    'perturbation_conc',
    'disease_condition',
    'organism',
    'modality',
    'partition',
    'version',
    'platform',
    'smiles',
    'experimental_technique',
    'intensity_image',
]


varlist = [
        'intensity_image', 'ROI_label', 'COV', 'COVERED_IMAGE_INTENSITY_RANGE', 
        'ENERGY', 'ENTROPY', 'EXCESS_KURTOSIS', 'HYPERFLATNESS', 'HYPERSKEWNESS', 
        'INTEGRATED_INTENSITY', 'INTERQUARTILE_RANGE', 'KURTOSIS', 'MAX', 'MEAN', 
        'MEAN_ABSOLUTE_DEVIATION', 'MEDIAN', 'MEDIAN_ABSOLUTE_DEVIATION', 'MIN', 'MODE',
        'P01', 'P10', 'P25', 'P75', 'P90', 'P99', 'QCOD', 'RANGE', 'ROBUST_MEAN', 
        'ROBUST_MEAN_ABSOLUTE_DEVIATION', 'ROOT_MEAN_SQUARED', 'SKEWNESS', 'STANDARD_DEVIATION', 
        'STANDARD_DEVIATION_BIASED', 'STANDARD_ERROR', 'VARIANCE', 'VARIANCE_BIASED', 'UNIFORMITY',
        'UNIFORMITY_PIU', 'GLCM_ASM_0', 'GLCM_ASM_45', 'GLCM_ASM_90', 'GLCM_ASM_135', 'GLCM_ACOR_0',
        'GLCM_ACOR_45', 'GLCM_ACOR_90', 'GLCM_ACOR_135', 'GLCM_CLUPROM_0', 'GLCM_CLUPROM_45', 'GLCM_CLUPROM_90', 
        'GLCM_CLUPROM_135', 'GLCM_CLUSHADE_0', 'GLCM_CLUSHADE_45', 'GLCM_CLUSHADE_90', 'GLCM_CLUSHADE_135', 
        'GLCM_CLUTEND_0', 'GLCM_CLUTEND_45', 'GLCM_CLUTEND_90', 'GLCM_CLUTEND_135', 'GLCM_CONTRAST_0', 
        'GLCM_CONTRAST_45', 'GLCM_CONTRAST_90', 'GLCM_CONTRAST_135', 'GLCM_CORRELATION_0', 'GLCM_CORRELATION_45',
        'GLCM_CORRELATION_90', 'GLCM_CORRELATION_135', 'GLCM_DIFAVE_0', 'GLCM_DIFAVE_45', 'GLCM_DIFAVE_90', 
        'GLCM_DIFAVE_135', 'GLCM_DIFENTRO_0', 'GLCM_DIFENTRO_45', 'GLCM_DIFENTRO_90', 'GLCM_DIFENTRO_135', 
        'GLCM_DIFVAR_0', 'GLCM_DIFVAR_45', 'GLCM_DIFVAR_90', 'GLCM_DIFVAR_135', 'GLCM_DIS_0', 'GLCM_DIS_45', 
        'GLCM_DIS_90', 'GLCM_DIS_135', 'GLCM_ENERGY_0', 'GLCM_ENERGY_45', 'GLCM_ENERGY_90', 'GLCM_ENERGY_135', 
        'GLCM_ENTROPY_0', 'GLCM_ENTROPY_45', 'GLCM_ENTROPY_90', 'GLCM_ENTROPY_135', 'GLCM_HOM1_0', 'GLCM_HOM1_45', 
        'GLCM_HOM1_90', 'GLCM_HOM1_135', 'GLCM_HOM2_0', 'GLCM_HOM2_45', 'GLCM_HOM2_90', 'GLCM_HOM2_135', 
        'GLCM_ID_0', 'GLCM_ID_45', 'GLCM_ID_90', 'GLCM_ID_135', 'GLCM_IDN_0', 'GLCM_IDN_45', 'GLCM_IDN_90',
        'GLCM_IDN_135', 'GLCM_IDM_0', 'GLCM_IDM_45', 'GLCM_IDM_90', 'GLCM_IDM_135', 'GLCM_IDMN_0', 'GLCM_IDMN_45',
        'GLCM_IDMN_90', 'GLCM_IDMN_135', 'GLCM_INFOMEAS1_0', 'GLCM_INFOMEAS1_45', 'GLCM_INFOMEAS1_90',
        'GLCM_INFOMEAS1_135', 'GLCM_INFOMEAS2_0', 'GLCM_INFOMEAS2_45', 'GLCM_INFOMEAS2_90', 'GLCM_INFOMEAS2_135', 
        'GLCM_IV_0', 'GLCM_IV_45', 'GLCM_IV_90', 'GLCM_IV_135', 'GLCM_JAVE_0', 'GLCM_JAVE_45', 'GLCM_JAVE_90', 
        'GLCM_JAVE_135', 'GLCM_JE_0', 'GLCM_JE_45', 'GLCM_JE_90', 'GLCM_JE_135', 'GLCM_JMAX_0', 'GLCM_JMAX_45', 
        'GLCM_JMAX_90', 'GLCM_JMAX_135', 'GLCM_JVAR_0', 'GLCM_JVAR_45', 'GLCM_JVAR_90', 'GLCM_JVAR_135', 
        'GLCM_SUMAVERAGE_0', 'GLCM_SUMAVERAGE_45', 'GLCM_SUMAVERAGE_90', 'GLCM_SUMAVERAGE_135', 'GLCM_SUMENTROPY_0', 
        'GLCM_SUMENTROPY_45', 'GLCM_SUMENTROPY_90', 'GLCM_SUMENTROPY_135', 'GLCM_SUMVARIANCE_0', 'GLCM_SUMVARIANCE_45', 
        'GLCM_SUMVARIANCE_90', 'GLCM_SUMVARIANCE_135', 'GLCM_VARIANCE_0', 'GLCM_VARIANCE_45', 'GLCM_VARIANCE_90', 
        'GLCM_VARIANCE_135', 'GLCM_ASM_AVE', 'GLCM_ACOR_AVE', 'GLCM_CLUPROM_AVE', 'GLCM_CLUSHADE_AVE', 
        'GLCM_CLUTEND_AVE', 'GLCM_CONTRAST_AVE', 'GLCM_CORRELATION_AVE', 'GLCM_DIFAVE_AVE', 'GLCM_DIFENTRO_AVE', 
        'GLCM_DIFVAR_AVE', 'GLCM_DIS_AVE', 'GLCM_ENERGY_AVE', 'GLCM_ENTROPY_AVE', 'GLCM_HOM1_AVE', 'GLCM_ID_AVE', 
        'GLCM_IDN_AVE', 'GLCM_IDM_AVE', 'GLCM_IDMN_AVE', 'GLCM_IV_AVE', 'GLCM_JAVE_AVE', 'GLCM_JE_AVE', 
        'GLCM_INFOMEAS1_AVE', 'GLCM_INFOMEAS2_AVE', 'GLCM_VARIANCE_AVE', 'GLCM_JMAX_AVE', 'GLCM_JVAR_AVE', 
        'GLCM_SUMAVERAGE_AVE', 'GLCM_SUMENTROPY_AVE', 'GLCM_SUMVARIANCE_AVE', 'FRAC_AT_D_0', 'FRAC_AT_D_1', 
        'FRAC_AT_D_2', 'FRAC_AT_D_3', 'FRAC_AT_D_4', 'FRAC_AT_D_5', 'FRAC_AT_D_6', 'FRAC_AT_D_7', 'GABOR_0', 
        'GABOR_1', 'GABOR_2', 'GABOR_3', 'MEAN_FRAC_0', 'MEAN_FRAC_1', 'MEAN_FRAC_2', 'MEAN_FRAC_3', 'MEAN_FRAC_4',
        'MEAN_FRAC_5', 'MEAN_FRAC_6', 'MEAN_FRAC_7', 'RADIAL_CV_0', 'RADIAL_CV_1', 'RADIAL_CV_2', 'RADIAL_CV_3', 
        'RADIAL_CV_4', 'RADIAL_CV_5', 'RADIAL_CV_6', 'RADIAL_CV_7', 'ZERNIKE2D_Z0', 'ZERNIKE2D_Z1', 'ZERNIKE2D_Z2',
        'ZERNIKE2D_Z3', 'ZERNIKE2D_Z4', 'ZERNIKE2D_Z5', 'ZERNIKE2D_Z6', 'ZERNIKE2D_Z7', 'ZERNIKE2D_Z8', 'ZERNIKE2D_Z9', 
        'ZERNIKE2D_Z10', 'ZERNIKE2D_Z11', 'ZERNIKE2D_Z12', 'ZERNIKE2D_Z13', 'ZERNIKE2D_Z14', 'ZERNIKE2D_Z15',
        'ZERNIKE2D_Z16', 'ZERNIKE2D_Z17', 'ZERNIKE2D_Z18', 'ZERNIKE2D_Z19', 'ZERNIKE2D_Z20', 'ZERNIKE2D_Z21',
        'ZERNIKE2D_Z22', 'ZERNIKE2D_Z23', 'ZERNIKE2D_Z24', 'ZERNIKE2D_Z25', 'ZERNIKE2D_Z26', 'ZERNIKE2D_Z27', 
        'ZERNIKE2D_Z28', 'ZERNIKE2D_Z29', 'SPAT_MOMENT_00', 'SPAT_MOMENT_01', 'SPAT_MOMENT_02', 'SPAT_MOMENT_03',
        'SPAT_MOMENT_10', 'SPAT_MOMENT_11', 'SPAT_MOMENT_12', 'SPAT_MOMENT_13', 'SPAT_MOMENT_20', 'SPAT_MOMENT_21', 
        'SPAT_MOMENT_22', 'SPAT_MOMENT_23', 'SPAT_MOMENT_30', 'CENTRAL_MOMENT_00', 'CENTRAL_MOMENT_01', 
        'CENTRAL_MOMENT_02', 'CENTRAL_MOMENT_03', 'CENTRAL_MOMENT_10', 'CENTRAL_MOMENT_11', 'CENTRAL_MOMENT_12', 
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
        ]



def link_medatadata(inp, metapath):
  meta = vaex.open(metapath)
  meta["plate"] = meta["plate"].astype(str)
  meta["site"] = meta["site"].astype(str)
  df = vaex.open(inp)
  images = df["intensity_image"].tolist()
  pattern = re.compile(r"(?P<well>\w+)_s(?P<site>\d+)_w(?P<channel>\d).*.ome.tif")
  match = [pattern.match(i) for i in images]
  well = [r.group("well") for r in match]
  site = [r.group("site") for r in match]
  wellnumber = [''.join(re.findall(r'\d', w)) for w in well]
  channelnumber = [r.group("channel") for r in match]
  df['well'] = np.array(well)
  df['site'] = np.array(site)
  df['wellnumber'] = np.array(wellnumber)
  df['channelnumber'] = np.array(channelnumber)
  df['z_position'] = np.array([None] * len(df))
  df['perturbation_id'] = np.array([None] * len(df))
  df['partition'] = np.array([None] * len(df))
  df['version'] = np.array([None] * len(df))
  df['platform'] = np.array([None] * len(df))
  df['channel'] = df['channelnumber'].map(channel_dict)
  df['control_type'] = np.array([None] * len(df))
  df['smiles'] = np.array([None] * len(df))
  df['disease_condition'] = np.array([None] * len(df))
  df['organism'] = np.array(['human'] * len(df))
  df['modality'] = np.array(["Optical Imaging"] * len(df))
  df['experimental_technique'] = np.array(["Morphological Imaging Dataset of immune perturbations"] * len(df))


# Concatenate columns into a composite key
  meta['composite_key'] = meta['experiment'] + '_' + meta['plate'] + '_' + meta['well'] + '_' + meta['site']
  df['composite_key'] = df['experiment'] + '_' + df['plate'] + '_' + df['well']+ '_' + df['site']

  combined = df.join(meta, left_on='composite_key', right_on='composite_key', how='inner', lsuffix='_left', rsuffix='_right', allow_duplication=True)
  columns = combined.get_column_names()

  # Create a dictionary for renaming columns
  rename_dict = {col: col.replace('_right', '') for col in columns if col.endswith('_right')}
  rename_cols = {"cell_type": "cell",
                "treatment": "perturbation",
              "treatment_conc": "perturbation_conc",
                }
  rename_dict.update(rename_cols)

  for item in rename_dict.items():
      oldname, newname = item
      combined.rename(oldname, newname)

  remove_col = [col for col in columns if re.search('.*_left', col)]
  exclude_col = ["site_id", "well_id", "composite_key"] + remove_col
  combined.drop(columns=exclude_col, inplace=True)

  final_columns = metacols + varlist
  combined = combined[final_columns]

  return combined



dataset = "rxrx2"
inp_dir = f'/projects/PanMicroscopy/NyxusFeatures/recursionpharma/combined_arrow/{dataset}'
metapath = f'/projects/PanMicroscopy/data/recursionpharma/raw/{dataset}/{dataset}-metadata/metadata.csv'


pattern = os.path.join(inp_dir, '**', '*.arrow')
arrow_files = glob.glob(pattern, recursive=True)


for inp in tqdm(arrow_files, desc="Processing", leave=True, mininterval=0.1, total=len(arrow_files)):
  time.sleep(0.01) 
  st = time.time()

  prf = link_medatadata(inp, metapath)


  output_path = f"{inp_dir}/combine_arrowfile"
  logger.info(f'writing combined file {output_path}')
  if not Path(output_path).exists():
      Path(output_path).mkdir(parents=True, exist_ok=True)
  output_file = f"{output_path}/rxrx2_NyxusFeatures.arrow"


        # Handle the append logic manually
  if Path(output_file).exists():
      # If the file exists, load it and concatenate the new chunk
      existing_df = vaex.open(output_file)
      combined_df = vaex.concat([existing_df, prf])
      combined_df.export_feather(output_file)
  else:
      # If the file does not exist, export the chunk as a new file
      prf.export_feather(output_file)

  del prf
  gc.collect()


  logger.info(f"combined: {inp} arrow file")


final = (time.time() - st)/60
logger.info(f"Total time taken: {final} min")
