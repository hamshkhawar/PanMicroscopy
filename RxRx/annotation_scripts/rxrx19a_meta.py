from pathlib import Path
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.dataset as ds
import pandas as pd
import re
import time
import gc
import warnings

warnings.filterwarnings("ignore")

starttime = time.time()

dataset = "rxrx19a"
inp_dir = Path('/projects/PanMicroscopy/NyxusFeatures/recursionpharma')
out_dir = Path("/projects/PanMicroscopy/NyxusFeatures/recursionpharma/RxRx_combined/rxrx19a")
metapath = f'/projects/PanMicroscopy/data/recursionpharma/raw/{dataset}/{dataset}-metadata/metadata.csv'

inp_dir = inp_dir.joinpath(dataset)
arrowpaths = [path for path in inp_dir.rglob('*') if path.is_file() and path.name == "NyxusFeatures.arrow"]

# Metadata loading
meta = csv.read_csv(metapath).to_pandas()
meta["plate"] = meta["plate"].astype(str)
meta["site"] = meta["site"].astype(str)
meta['composite_key'] = meta['experiment'] + '_' + meta['plate'] + '_' + meta['well'] + '_' + meta['site']

channel_dict = {
    '1': 'Hoechst-33342',
    '2': 'Concanavalin A',
    '3': 'Phalloidin',
    '4': 'Syto14',
    '5': 'WGA'
}

disease_dict = {
    'Active SARS-CoV-2': 'Active Virus',
    'UV Inactivated SARS-CoV-2': 'Inactivated Virus',
    '': 'Unspecified',
    'Mock':'Control'
}


# Ensure output directory exists
out_dir.mkdir(parents=True, exist_ok=True)

# Processing each file
for i, p in enumerate(arrowpaths):
    print(p)
    image_path = str(p.parent).replace("NyxusFeatures", "data")
    image_path = image_path.replace(f"{dataset}", f"omeconverted/{dataset}")
    experiment = p.parents[1].name
    plate = re.findall(r'\d+', p.parents[0].name)[0]
    
    # Open the Arrow file
    table = pa.ipc.open_file(p).read_all()

    images = table.column("intensity_image").to_pylist()
    pattern = re.compile(r"(?P<well>\w+)_s(?P<site>\d+)_w(?P<channel>\d).*.ome.tif")
    match = [pattern.match(i) for i in images]
    well = [r.group("well") for r in match]
    wellnumber = [''.join(re.findall(r'\d+', w)) for w in well]
    site = [r.group("site") for r in match]
    channelnumber = [r.group("channel") for r in match]

    # Create new columns
    new_columns = {
        'path': pa.array([f"{image_path}/{img}" for img in images]),
        'dataset': pa.array([dataset] * len(table)),
        'experiment': pa.array([experiment] * len(table)),
        'plate': pa.array([plate] * len(table)),
        'well': pa.array(well),
        'site': pa.array(site),
        'channelnumber': pa.array(channelnumber),
        'wellnumber': pa.array(wellnumber),
    }
    for col, data in new_columns.items():
        table = table.append_column(col, data)
    
    # Convert to pandas for merging with metadata
    combined_pandas = table.to_pandas()
    combined_pandas['composite_key'] = combined_pandas['experiment'] + '_' + combined_pandas['plate'] + '_' + combined_pandas['well'] + '_' + combined_pandas['site']
    combined_pandas = combined_pandas.merge(meta, on='composite_key', how='inner')

    # Add final columns
    combined_pandas['channelname'] = combined_pandas['channelnumber'].map(channel_dict)
    combined_pandas['organism'] = 'human'
    combined_pandas['z_position'] = None
    combined_pandas['perturbation_id'] = None
    combined_pandas['partition'] = None
    combined_pandas['version'] = None
    combined_pandas['control_type'] = combined_pandas['disease_condition'].map(disease_dict)
    combined_pandas['modality'] = 'Optical Imaging'
    combined_pandas['experimental_technique'] = 'high-dimensional human cellular assay for COVID-19 associated disease'

    combined_pandas.drop(columns=['experiment_y', 'plate_y', 'well_y', 'site_y', 'composite_key', 'site_id', 'well_id',], inplace=True)

   
    combined_pandas.rename(columns={'experiment_x': 'experiment', 
                                                      'plate_x': 'plate',
                                                      'well_x': 'well',
                                                      'site_x': 'site',
                                                      'cell_type':'cell',
                                                      'treatment':'perturbation',
                                                      'treatment_conc': 'dose',
                                                      'SMILES' : 'smiles'

                                                      }, inplace=True)
    
    newcolumns =  list(combined_pandas.columns)[473:] + list(combined_pandas.columns)[:473]
    combined_pandas = combined_pandas[newcolumns]

  
    # Convert back to Arrow table
    combined_table = pa.Table.from_pandas(combined_pandas)
    
    # Write the combined table to a new file in the output directory
    output_file = out_dir / f"{experiment}_pl_{plate}_{i}.arrow"
    with pa.OSFile(str(output_file), 'wb') as f:
        with pa.ipc.new_file(f, combined_table.schema) as writer:
            writer.write(combined_table)
    
    del combined_table
    gc.collect()

print("Processing complete.")
finaltime = (time.time() - starttime) / 60
print(f"Total time taken in minutes: {finaltime}")


