from pathlib import Path
import vaex
import re
import numpy as np
import pandas as pd

df = vaex.open("/projects/PanMicroscopy/NyxusFeatures/recursionpharma/combined_arrow/rxrx19b/rxrx19b_NyxusFeatures.arrow")
pl1 = vaex.open("/projects/PanMicroscopy/NyxusFeatures/recursionpharma/combined_arrow/rxrx19b/HUVEC-1/Plate1/NyxusFeatures.arrow")
meta = vaex.open("/projects/PanMicroscopy/data/recursionpharma/raw/rxrx19b/rxrx19b-metadata/metadata.csv")
prf = vaex.concat([pl1, df])
meta["plate"] = meta["plate"].astype(str)


images = prf["intensity_image"].tolist()
pattern = re.compile("(?P<well>\w+)_s(?P<site>\d+)_w(?P<channel>\d).*.ome.tif")
match = [pattern.match(i) for i in images]
well= [r.group("well") for r in match]
wellnumber = [''.join(re.findall(r'\d', w)) for w in well]
channelnumber = [r.group("channel") for r in match]


channel_dict = {
    '1': 'Hoechst-33342',
    '2': 'Concanavalin A', 
    '3': 'Phalloidin', 
    '4': 'Syto14', 
    '5': 'MitoTracker', 
    '6': 'WGA'
}


prf['well'] =  np.array(well)
prf['wellnumber'] =  np.array(wellnumber)
prf['channelnumber'] =  np.array(channelnumber)
prf['z_position'] =  np.array([None] * len(prf))
prf['perturbation_id'] =  np.array([None] * len(prf))
prf['partition'] =  np.array([None] * len(prf))
prf['version'] =  np.array([None] * len(prf))
prf['platform'] =  np.array([None] * len(prf))
prf['channel'] = prf['channelnumber'].map(channel_dict)
prf['control_type'] = np.array([None] * len(prf))
prf['organism'] =  np.array(['human'] * len(prf))
prf['modality'] =  np.array(["Optical Imaging"] * len(prf))
prf['experimental_technique'] =  np.array(["high-dimensional human cellular assay for COVID-19 associated disease"] * len(prf))
prf = prf.to_pandas_df()
meta = meta.to_pandas_df()
merged_df = pd.merge(meta, prf, on=['experiment', 'plate', 'well'], how='inner')
merged_df.drop(columns=['site_id', 'well_id'], inplace=True)
merged_df.rename(columns={'treatment': 'perturbation', 'treatment_conc': 'perturbation_conc', 'SMILES': 'smiles', 'cell_type':'cell'}, inplace=True)


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

varcol = list(filter(lambda s: s[0].isupper(), list(merged_df.columns)))


combined = merged_df[metacols+varcol]

vaex_combined = vaex.from_pandas(combined)

output_file = Path('/projects/PanMicroscopy/NyxusFeatures/recursionpharma/combined_arrow/rxrx19b/rxrx19b_NyxusFeatures.arrow')
vaex_combined.export_feather(output_file)


