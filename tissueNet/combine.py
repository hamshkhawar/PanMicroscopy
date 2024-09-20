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

        x['tissue'] = np.array(tissue)
        x['platform'] = np.array(platform)
        x['channel'] = np.array(channel)

        # Flip the dictionary
        tissue_dict = {value: key for key, value in tissue_dict.items()}
        platform_dict = {value: key for key, value in platform_dict.items()}
        x['tissue'] = x['tissue'].map(tissue_dict)
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

        x['tissue'] = np.array(tissue)
        none_values = np.array([None] * len(x))
        x['platform'] =  none_values
        x['channel'] = np.array(channel)


        # Flip the dictionary
        tissue_dict = {value: key for key, value in tissue_dict.items()}
        x['tissue'] = x['tissue'].map(tissue_dict)


    
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

    for inp in inp_dir.rglob("*.arrow"):
        version = inp.parents[2].name
        if version == "v1.0":
            dirname = inp.parents[0].name
            df = vaex.open(inp)
            df['version'] = np.repeat(version, df.shape[0])
            df['dataset'] = np.repeat(dirname, df.shape[0])
            df_v10.append(df)

        if version == "v1.1":
            dirname = inp.parents[0].name
            df = vaex.open(inp)
            df['version'] = np.repeat(version, df.shape[0])
            df['dataset'] = np.repeat(dirname, df.shape[0])
            df_v11.append(df)

        
    df_v10 = vaex.concat(df_v10)
    df_v10 = metadata_extraction(df_v10, version="v1.0")

    logger.info(f'Combined features for dataset: v1.0')

    df_v11 = vaex.concat(df_v11)
    df_v11 = metadata_extraction(df_v11, version="v1.1")
    logger.info(f'Combined features for dataset: v1.1')

    df_concat = vaex.concat([df_v10, df_v11])

    output_file = out_dir.joinpath("combined_tissueNet.arrow")

    df_concat.export_feather(output_file)

    finishtime = (time.time() - starttime) / 60
    logger.info(f'total time taken in minutes {finishtime}')


if __name__ == '__main__':
    app()
