from pathlib import Path
import vaex
import re
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

dataset_name = "rxrx19a"
chunk_size = 1000  # Adjust this based on available memory

output_file = Path(f'/projects/PanMicroscopy/NyxusFeatures/recursionpharma/combined_arrow/{dataset_name}/combine_arrowfile/combined_{dataset_name}_NyxusFeatures.arrow')

# Load Metadata
meta = vaex.open(f"/projects/PanMicroscopy/data/recursionpharma/raw/{dataset_name}/{dataset_name}-metadata/metadata.csv")
meta["plate"] = meta["plate"].astype(str)
meta = meta.to_pandas_df()

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

# Load PRF in chunks and process each chunk
prf = vaex.open(f"/projects/PanMicroscopy/NyxusFeatures/recursionpharma/combined_arrow/{dataset_name}/combine_arrowfile/{dataset_name}_NyxusFeatures.arrow")
total_rows = len(prf)
print(f"loaded data")
num_chunks = (total_rows + chunk_size - 1) // chunk_size

for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, total_rows)
    prf_chunk = prf[start:end]
    
    # Process chunk
    images = prf_chunk["intensity_image"].tolist()
    pattern = re.compile(r"(?P<well>\w+)_s(?P<site>\d+)_w(?P<channel>\d).*.ome.tif")
    match = [pattern.match(i) for i in images]
    well = [r.group("well") for r in match]
    wellnumber = [''.join(re.findall(r'\d', w)) for w in well]
    channelnumber = [r.group("channel") for r in match]

    prf_chunk['well'] = np.array(well)
    prf_chunk['wellnumber'] = np.array(wellnumber)
    prf_chunk['channelnumber'] = np.array(channelnumber)
    prf_chunk['z_position'] = np.array([None] * len(prf_chunk))
    prf_chunk['perturbation_id'] = np.array([None] * len(prf_chunk))
    prf_chunk['partition'] = np.array([None] * len(prf_chunk))
    prf_chunk['version'] = np.array([None] * len(prf_chunk))
    prf_chunk['platform'] = np.array([None] * len(prf_chunk))
    prf_chunk['channel'] = prf_chunk['channelnumber'].map(channel_dict)
    prf_chunk['control_type'] = np.array([None] * len(prf_chunk))
    prf_chunk['organism'] = np.array(['human'] * len(prf_chunk))
    prf_chunk['modality'] = np.array(["Optical Imaging"] * len(prf_chunk))
    prf_chunk['experimental_technique'] = np.array(["high-dimensional human cellular assay for COVID-19 associated disease"] * len(prf_chunk))

    # Convert to pandas DataFrame
    prf_chunk = prf_chunk.to_pandas_df()

    # Merge with metadata
    merged_chunk = pd.merge(meta, prf_chunk, on=['experiment', 'plate', 'well'], how='inner')
    merged_chunk.drop(columns=['site_id', 'well_id'], inplace=True)
    merged_chunk.rename(columns={'treatment': 'perturbation', 'treatment_conc': 'perturbation_conc', 'SMILES': 'smiles', 'cell_type': 'cell'}, inplace=True)
    
    # Filter columns
    varcol = list(filter(lambda s: s[0].isupper(), list(merged_chunk.columns)))
    combined_chunk = merged_chunk[metacols + varcol]

    # Convert to Vaex DataFrame
    vaex_chunk = vaex.from_pandas(combined_chunk)

    # Handle the append logic manually
    if output_file.exists():
        # If the file exists, load it and concatenate the new chunk
        existing_df = vaex.open(output_file)
        combined_df = vaex.concat([existing_df, vaex_chunk])
        combined_df.export_feather(output_file)
    else:
        # If the file does not exist, export the chunk as a new file
        vaex_chunk.export_feather(output_file)

    # Cleanup
    del prf_chunk, merged_chunk, combined_chunk, vaex_chunk
    gc.collect()

    print(f"Successfully processed and exported {chunk_idx} rows to '{num_chunks}'.")

