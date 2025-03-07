from nyxus import Nyxus
import time
from pathlib import Path
import typer
import logging
import os
import re


# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("RxRx1 dataset")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))



app = typer.Typer()

def nyxfun(inp_dir, out_dir):

    file_pattern=".*.tif"

    nyx = Nyxus(["*WHOLESLIDE*", "-*SGEOMOMS*", "-GABOR"])


    nyx.using_gpu(True)

    nyx_params = {
        "neighbor_distance": 5,
        "pixels_per_micron": 1.0,
        "n_feature_calc_threads": 8,
    }

    nyx.set_params(**nyx_params)

    
    nyx.featurize_directory(intensity_dir=str(inp_dir), 
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
    )
    ):

    starttime = time.time()

    out_dir = re.sub(r"/omeconverted", "", str(inp_dir)).replace("data", "NyxusFeatures")

    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"output directory: {out_dir} created")

    nyxfun(inp_dir, out_dir)

    logger.info(f"finished featurization of plate: {inp_dir}")

    end = (time.time() - starttime)/60
    logger.info(f"Time taken: {end} min")

    
if __name__ == '__main__':
    app()





