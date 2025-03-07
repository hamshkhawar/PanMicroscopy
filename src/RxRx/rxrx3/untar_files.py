from pathlib import Path
import re
import numpy as np
from tqdm import tqdm
import typer
import time
import logging
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
app = typer.Typer()
import glob
import os
import tarfile
import subprocess
from nyxus import Nyxus



logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("Untar files")
logger.setLevel(logging.INFO)
num_workers = max([cpu_count() , 2])



def omeconverter(inp_dir:Path, file_pattern:str, file_extension:str, out_dir:Path) -> None:  
    """ Ome Converter 
    """
    command =  f"singularity run --bind /projects/PanMicroscopy:/projects/PanMicroscopy /home/abbasih2/plugins/ome-converter-tool_0_1_0_dev0.sif --inpDir={inp_dir} --filePattern={file_pattern} --fileExtension={file_extension} --outDir={out_dir}"
    # command =  f"singularity run --bind /projects/PanMicroscopy:/projects/PanMicroscopy --fakeroot /home/abbasih2/plugins/ome-converter-plugin_0_3_2_dev1.sif --env USER=polusai --userns --uid 1001 --inpDir={inp_dir} --filePattern={file_pattern} --fileExtension={file_extension} --outDir={out_dir}"

    _ = subprocess.run(command,shell=True,stdout=subprocess.PIPE)


    return 


def nyxfun(intensity_dir, file_pattern, out_dir, pixels_per_micron):

    nyx = Nyxus(["*ALL*"])

    nyx_params = {
        "neighbor_distance": 5,
        "pixels_per_micron": pixels_per_micron,
        # "n_feature_calc_threads": 8,
    }

    nyx.set_params(**nyx_params)

    
    nyx.featurize_directory(intensity_dir=str(intensity_dir), 
                            label_dir=None,
                            file_pattern=file_pattern,
                            output_type = "arrowipc",
                            output_path = str(out_dir))
    

def untar_file(tar_file):
    outpath = f"{Path(tar_file).parent}/{Path(tar_file).stem}"
    print(outpath)
    # if not Path(outpath).exists():
    #     Path(outpath).mkdir(parents=True, exist_ok=True)
  
    # with tarfile.open(tar_file, "r:*") as tar:
    #     tar.extractall(path=outpath)
    #     logger.info(f"Extracted '{tar_file}' to '{outpath}'")

    # # os.remove(tar_file)
    # logger.info(f"Removed tar file '{tar_file}'")

    # omeoutdir = Path("/projects/PanMicroscopy/data/recursionpharma/omeconverted/rxrx3")
    # if not Path(omeoutdir).exists():
    #     Path(outpath).mkdir(parents=True, exist_ok=True)




@app.command()
def main(
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        help="Untar image files",
        exists=True,
        resolve_path=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
    )
    ):

    starttime = time.time()

    pattern = os.path.join(inp_dir, '**', '*.tar')

    tar_files = glob.glob(pattern, recursive=True)

    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        threads = [exe.submit(untar_file, tar_file) for tar_file in tar_files]

        for f in tqdm(
            as_completed(threads),
            total=len(threads),
            mininterval=1,
            desc=f"Untar plate",
            initial=0,
            unit_scale=True,
            colour="cyan",
        ):
            try:
            # Wait for the result of the future and print it
                f.result()  # This will also raise exceptions if any
                print(f"Thread {f} has completed successfully.")
            except Exception as e:
                # Print out an error message if an exception occurred
                print(f"Thread {f} encountered an error: {e}")


   
    finishtime = (time.time() - starttime) / 60
    logger.info(f'total time taken in minutes {finishtime}')
 
     


if __name__ == '__main__':
    app()
