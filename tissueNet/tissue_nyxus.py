
from pathlib import Path
from multiprocessing import cpu_count
import filepattern as fp
from tqdm import tqdm 
import os
from nyxus import Nyxus
import time
import logging
import preadator
import typer
from concurrent.futures import as_completed

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("Recursion dataset")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

NUM_THREADS = max(cpu_count() // 2, 2)

def split_path_at_string(path, target):
    parts = path.parts
    try:
        target_index = parts.index(target)
        before_target = Path(*parts[:target_index])
        after_target = Path(*parts[target_index + 1:-1])
        return before_target, after_target
    except ValueError:
        return None, None  # target not found in the path
    

def nyxfun(intensity_dir, file_pattern, out_dir):

    nyx = Nyxus(["*ALL*"])

    nyx_params = {
        "neighbor_distance": 5,
        "pixels_per_micron": 1.0,
        "n_feature_calc_threads": 28,
    }

    nyx.set_params(**nyx_params)

    
    nyx.featurize_directory(intensity_dir=str(intensity_dir), 
                            label_dir=None,
                            file_pattern=file_pattern,
                            output_type = "arrowipc",
                            output_path = str(out_dir))
                            
    
def featextraction(intensity_dir, out_dir):

    fps = fp.FilePattern(intensity_dir, ".*.ome.tif")


    _, target = split_path_at_string(intensity_dir, "tissueNet")

    out_dir = Path(out_dir, target)
    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True, parents=True)

    nyxfun(intensity_dir=intensity_dir, file_pattern=".*", out_dir=out_dir)


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

    for d in list(inp_dir.iterdir())[1:]:

        folderpath = [path for path in d.rglob('*') if path.is_dir() and "intensity"  in path.name]

        with preadator.ProcessManager(
            name="Extracting nyxus features",
            num_processes=NUM_THREADS,
            threads_per_process=2,
        ) as executor:
            threads = []
            for f in folderpath:
                threads.append(
                    executor.submit(featextraction, f, out_dir),
                )

            for f in tqdm(
                as_completed(threads),
                total=len(threads),
                mininterval=5,
                desc=f"Extracting features {f}",
                initial=0,
                unit_scale=True,
                colour="cyan",
            ):
                f.result()

        finishtime = (time.time() - starttime) / 60
        logger.info(f'total time taken in minutes {finishtime}')


if __name__ == '__main__':
    app()
