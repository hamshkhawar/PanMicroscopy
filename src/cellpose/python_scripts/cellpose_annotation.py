from pathlib import Path
import vaex
import numpy as np
import re


dataset = "cellpose"
inp_dir = Path('/projects/PanMicroscopy/NyxusFeatures/')
out_dir=Path("/projects/PanMicroscopy/NyxusFeatures/cellpose_combined")
path = f'/projects/PanMicroscopy/data/{dataset}/omeconverted'

inp_dir = inp_dir.joinpath(dataset)
arrowpaths = [path for path in inp_dir.rglob('*') if path.is_file() if path.name == "NyxusFeatures.arrow"]


x = []
for p in arrowpaths:
    image_path = str(p.parent).replace("NyxusFeatures", "data")
    image_path = image_path.replace(f"{dataset}", f"{dataset}/omeconverted")
    experiment=p.parent.name
    x_df = vaex.open(p)
    x_df['path'] =  np.array([image_path] * len(x_df))
    x_df['path'] = x_df['path'] + '/' + x_df['intensity_image']
    x_df['dataset'] =  np.array([dataset] * len(x_df))
    x_df['experiment'] =  np.array([experiment] * len(x_df))
    x_df['plate'] =  np.array([None] * len(x_df))
    x.append(x_df)

x = vaex.concat(x)

images = x['intensity_image'].tolist()

pattern = re.compile(".*img_c(?P<channelnumber>\d)")
match = [pattern.match(i) for i in images]
channelnumber = [r.group("channelnumber") for r in match]


x['well'] =  np.array([None] * len(x))
x['site'] = np.array([None] * len(x))
x['wellnumber'] =  np.array([None] * len(x))
x['channelnumber'] = np.array(channelnumber)
x['z_position'] =  np.array([None] * len(x))
x['perturbation_id'] = np.array([None] * len(x))
x['partition'] = np.array([None] * len(x))
x['version'] = np.array([None] * len(x))
x['platform'] = np.array([None] * len(x))
x['channelname'] = np.array([None] * len(x))
x['control_type'] =np.array([None] * len(x))
x['dose'] = np.array([None]* len(x))
x['smiles'] = np.array([None] * len(x))
x['disease_condition'] = np.array([None] * len(x))
x['organism'] = np.array([None] * len(x))
x['cell'] =  np.array([None] * len(x))
x['modality'] = np.array(["Optical Imaging"] * len(x))
x['experimental_technique'] = np.array(["Cellpose dataset for cellular segmentation training"] * len(x))

if not Path(out_dir).exists():
    Path(out_dir).mkdir(exist_ok=True, parents=True)

x.export_feather(out_dir.joinpath(f"{dataset}.arrow"))