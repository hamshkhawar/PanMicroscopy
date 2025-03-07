import vaex
from pathlib import Path
import numpy as np
import re
import os
import logging
import typer
import time

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("Combine tissueNet dataset")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


def metadata_extraction(x, version):

    images = x['intensity_image'].tolist()
 
    if version == "v1.0":

        tissue_dict = {'breast':"0", 'gi':"1", 'immune':"2", 'lung':"3", 'pancreas':"4", 'skin':"5"}
        platform_dict = {'codex': "0", 'cycif':"1", 'imc':"2", 'mibi':"3", 'mxif':"4", 'vectra':"5"} 
        pattern = re.compile("p(?P<platform>\d+)_y(?P<tissue>\d+)_r(?P<index>\d+)_c(?P<channel>\d+)")
        match = [pattern.match(i) for i in images]
        platform = [r.group("platform") for r in match]
        tissue = [r.group("tissue") for r in match]
        channel = [r.group("channel") for r in match]

        x['cell'] = np.array(tissue)
        x['platform'] = np.array(platform)
        x['channelnumber'] = np.array(channel)

        # Flip the dictionary
        tissue_dict = {value: key for key, value in tissue_dict.items()}
        platform_dict = {value: key for key, value in platform_dict.items()}
        x['cell'] = x['cell'].map(tissue_dict)
        x['platform'] = (x['platform']
                        .map(platform_dict)
                        )

    if version == "v1.1":

        tissue_dict = {
            "breast": "0",
            "colon": "1",
            "lymph node": "2",
            "lung": "3",
            "pancreas": "4",
            "epidermis": "5",
            "esophagus": "6",
            "spleen": "7",
            "tonsil": "8",
            "lymph node metastasis": "9",
        }

        pattern = re.compile("y(?P<tissue>\d+)_r(?P<index>\d+)_c(?P<channel>\d+)")
        match = [pattern.match(i) for i in images]
        tissue = [r.group("tissue") for r in match]
        channel = [r.group("channel") for r in match]

        x['cell'] = np.array(tissue)
        none_values = np.array([None] * len(x))
        x['platform'] =  none_values
        x['channelnumber'] = np.array(channel)


        # Flip the dictionary
        tissue_dict = {value: key for key, value in tissue_dict.items()}
        x['cell'] = x['cell'].map(tissue_dict)


    
    return x


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


    df_v10 = []
    df_v11 = []
    dataset="tissueNet"
    for inp in inp_dir.rglob("*.arrow"):
        version = inp.parents[3].name
        if version == "v1.0":
            image_path = str(inp.parent).replace("NyxusFeatures", "data")
            experiment=inp.parents[1].name
            x_df = vaex.open(inp)
            x_df['path'] =  np.array([image_path] * len(x_df))
            x_df['path'] = x_df['path'] + '/' + x_df['intensity_image ']
            x_df['dataset'] =  np.array([dataset] * len(x_df))
            x_df['experiment'] =  np.array([experiment] * len(x_df))
            x_df['plate'] =  np.array([None] * len(x_df))
            x_df['version'] = np.array([version] * len(x_df))
            df_v10.append(x_df)

        if version == "v1.1":
            image_path = str(inp.parent).replace("NyxusFeatures", "data")
            experiment=inp.parents[1].name
            x_df = vaex.open(inp)
            x_df['path'] =  np.array([image_path] * len(x_df))
            x_df['path'] = x_df['path'] + '/' + x_df['intensity_image ']
            x_df['dataset'] =  np.array([dataset] * len(x_df))
            x_df['experiment'] =  np.array([experiment] * len(x_df))
            x_df['plate'] =  np.array([None] * len(x_df))
            x_df['version'] = np.array([version] * len(x_df))
            df_v11.append(x_df)

        
    df_v10 = vaex.concat(df_v10)
    df_v10 = metadata_extraction(df_v10, version="v1.0")

    logger.info(f'Combined features for dataset: v1.0')

    df_v11 = vaex.concat(df_v11)
    df_v11 = metadata_extraction(df_v11, version="v1.1")
    logger.info(f'Combined features for dataset: v1.1')

    x = vaex.concat([df_v10, df_v11])

    x['well'] =  np.array([None] * len(x))
    x['site'] = np.array([None] * len(x))
    x['wellnumber'] =  np.array([None] * len(x))
    x['channelnumber'] = np.array([None] * len(x))
    x['z_position'] =  np.array([None] * len(x))
    x['perturbation_id'] = np.array([None] * len(x))
    x['partition'] = np.array([None] * len(x))
    x['version'] = np.array([None] * len(x))
    x['channelname'] = np.array([None]* len(x))
    x['control_type'] =np.array([None] * len(x))
    x['dose'] = np.array([None]* len(x))
    x['smiles'] = np.array([None] * len(x))
    x['disease_condition'] = np.array([None] * len(x))
    x['organism'] = np.array([None] * len(x))
    x['modality'] = np.array(["Optical Imaging"] * len(x))
    x['experimental_technique'] = np.array(["TissueNet dataset for training models on nuclear and whole cell segmentation in tissue images."] * len(x))

    metcols = ['path','dataset','experiment',
        'plate',
        'well',
        'site',
        'wellnumber',
        'channelnumber',
        'z_position',
        'perturbation_id',
        'partition',
        'version',
        'platform',
        'channelname',
        'control_type',
        'dose',
        'smiles',
        'disease_condition',
        'organism',
        'cell',
        'modality',
        'experimental_technique']

    newcolumns = x.column_names[:473] + metcols
    x =x[newcolumns]

    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    x.export_feather(out_dir.joinpath(f"{dataset}.arrow"))


    finishtime = (time.time() - starttime) / 60
    logger.info(f'total time taken in minutes {finishtime}')


if __name__ == '__main__':
    app()
