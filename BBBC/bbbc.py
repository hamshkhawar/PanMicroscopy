import sys
import time
import subprocess
from typing import Optional
from pathlib import Path
import logging
import numpy as np
import typer
from nyxus import Nyxus
from tqdm import tqdm
import preadator
from concurrent.futures import as_completed
import filepattern as fp
from multiprocessing import cpu_count
from bfio import BioReader
import vaex
import re
import shutil


app = typer.Typer()

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("Download bbbc dataset")
logger.setLevel(logging.INFO)


NUM_THREADS = max(cpu_count() // 2, 2)
   
def bbc_download() -> None:  
    """ 
    """
    logger.info('downloading')

    namelist = ["BBBC017", "BBBC021", "BBBC022"
    ]

    for name in namelist:
        command =  f"singularity run --bind /projects/PanMicroscopy:/projects/PanMicroscopy bbbc-download-plugin_0_1_0-dev1.sif --name={name} --outDir=/projects/PanMicroscopy/data"

        p = subprocess.run(command,shell=True,stdout=subprocess.PIPE)
    
    return 

# def file_renaming(name:str, file_pattern:str, out_file_pattern:str, out_dir:Path) -> None:  
#     """ File Renaming 
#     """

#     name = "BBBC001"
#     file_pattern = "/.*/Images/.*/.*_{row:c}{col:dd}f{f:dd}d{channel:d}.tif"
    
#     out_pattern="x{row:dd}_y{col:dd}_p{f:dd}_c{channel:d}.tif"

#     out_dir = Path('/projects/PanMicroscopy/data/processed_BBBC').joinpath(name)

#     if not Path(out_dir).exists():
#         Path(out_dir).mkdir(parents=True, exist_ok=False)
#         logger.info(f"{out_dir} is created")


#     command =  f"singularity run --bind /projects/PanMicroscopy:/projects/PanMicroscopy plugins/file-renaming-tool_0_2_4_dev2new2.sif --inpDir=/projects/PanMicroscopy/data/BBBC/{name} --filePattern={file_pattern} --outFilePattern={out_pattern} --outDir={out_dir}"

#     p = subprocess.run(command,shell=True,stdout=subprocess.PIPE)


#     return 

def max_path_parts(path):
    # Create a Path object
    path_obj = Path(path)
    
    # Get the parts of the path
    path_parts = path_obj.parts
    
    # Get the number of parts
    num_parts = len(path_parts)
    
    return num_parts

def file_renaming(inp_dir:Path, name:str, file_pattern:str, out_file_pattern:str, out_dir:Path) -> None:  
    """ File Renaming 
    """

    command =  f"singularity run --bind /projects/PanMicroscopy:/projects/PanMicroscopy /home/abbasih2/plugins/file-renaming-tool_0_2_4_dev2.sif --inpDir={inp_dir} --filePattern={file_pattern} --outFilePattern={out_file_pattern} --outDir={out_dir}"

    _ = subprocess.run(command,shell=True,stdout=subprocess.PIPE)


    return 



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
    

def feature_extraction(intensity_dir, file_pattern, out_dir):

    fps = fp.FilePattern(intensity_dir, file_pattern)
    print(intensity_dir)
    # flist = [f[1][0] for f in fps()]

    # logger.info("Processing BBBC dataset")


    # with preadator.ProcessManager(
    #     name="Extracting nyxus features",
    #     num_processes=NUM_THREADS,
    #     threads_per_process=20,
    # ) as executor:
    #     threads = []
    #     for f in flist:
    #         threads.append(
    #             executor.submit(nyxfun, f, file_pattern, out_dir),
    #         )

    #     for f in tqdm(
    #         as_completed(threads),
    #         total=len(threads),
    #         mininterval=5,
    #         desc=f"Extracting features {f}",
    #         initial=0,
    #         unit_scale=True,
    #         colour="cyan",
    #     ):
    #         f.result()
                            



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
    name: str = typer.Option(
        ...,
        "--name",
        help="Name of BBBC dataset",
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        help="Pattern",
    ),
    out_file_pattern: str = typer.Option(
        ".*",
        "--outFilePattern",
        help="output Pattern",
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

    inp_dir = inp_dir.joinpath(name)

    folderpath = sorted([path for path in inp_dir.rglob('*') if path.is_dir() if "Images" in path.parts and not "Images" in path.name])
    max_parts = max([max_path_parts(path) for path in inp_dir.rglob('*') if path.is_dir() and "Images" in path.parts])

    folderpath = sorted([path for path in inp_dir.rglob('*') if path.is_dir() and "Images" in path.parts if max_path_parts(path) == max_parts])


    # rename_dir = Path(inp_dir, 'rename')
    # if not Path(rename_dir).exists():
    #     Path(rename_dir).mkdir(parents=True, exist_ok=False)
    #     logger.info(f"{rename_dir} is created")

    # logger.info(f'Renaming images files of dataset: {name} --- Started')

    # file_renaming(inp_dir=inp_dir, name=name, file_pattern=file_pattern, out_file_pattern=out_file_pattern, out_dir=rename_dir)


    # logger.info(f' Renaming images files of dataset: {name} --- Completed!!!')

    if name in ["BBBC007", "BBBC020"]:
        for fl in folderpath:
            root_path = fl.parent.name
            rename = Path(f'{inp_dir}/rename/{root_path}')
            flname = re.sub(r'\W+', '_', fl.name)
            flname = flname.rstrip('_')
            rename_dir = Path(rename, flname)
            if not Path(rename_dir).exists():
                Path(rename_dir).mkdir(parents=True, exist_ok=False)

            files = fp.FilePattern(fl, file_pattern)

            for image in files():
                image = image[1][0]
                imagename = re.sub(r'[\s+\+|()|_|-]', '', image.name)
                newname = (Path(rename_dir, imagename))
                shutil.copy(image, newname)

    
        renamepath = sorted([path for path in rename.rglob('*') if path.is_dir()])
        for fl in renamepath:
            fps = fp.FilePattern(fl, file_pattern)
            flist = [f[1][0] for f in fps()]
            if len(flist) > 0:
                ome_dir = Path(inp_dir, 'omeconverted', fl.name)
                if not Path(ome_dir).exists():
                    Path(ome_dir).mkdir(parents=True, exist_ok=False)
                logger.info(f"{ome_dir} is created") 

                omeconverter(inp_dir=fl, file_pattern=file_pattern, file_extension=".ome.tif", out_dir=ome_dir)
                logger.info(f"Extracting nyxus feature {name}")

                nyxdir = Path(out_dir, name, fl.name)
                if not nyxdir.exists():
                    nyxdir.mkdir(exist_ok=True, parents=True)

            
                fps = fp.FilePattern(ome_dir, ".*.ome.tif")
                image_path = [f[1][0] for f in fps()]
                if len(image_path) > 0:
                    br = BioReader(image_path[0])
                    if br.physical_size_y[0] is not None:
                        pixels_per_micron = br.physical_size_y[0]
                    else:
                        pixels_per_micron = 1.0

                    logger.info(f'pixels_per_micro: {pixels_per_micron}')

                    nyxfun(intensity_dir=ome_dir, file_pattern=".*.ome.tif", out_dir=nyxdir, pixels_per_micron = pixels_per_micron)

                    arrowpath = Path(nyxdir, 'NyxusFeatures.arrow')

                    x = vaex.open(arrowpath)
                    x['dataset'] =  np.array([name] * len(x))
                    x['plate'] =  np.array([fl.name] * len(x))
                    x['path'] =  np.array([str(ome_dir)] * len(x))

                    finishtime = (time.time() - starttime) / 60
                    logger.info(f'total time taken in minutes {finishtime}')
                    x.export_feather(nyxdir.joinpath(f"combined_NyxusFeatures.arrow"))
                        
    else:
        for fl in folderpath:
            split_string = str(fl).split('Images')[-1].lstrip('/')
            fps = fp.FilePattern(fl, file_pattern)
            flist = [f[1][0] for f in fps()]
            if len(flist) > 0:
                ome_dir = Path(inp_dir, 'omeconverted', split_string)
                if not Path(ome_dir).exists():
                    Path(ome_dir ).mkdir(parents=True, exist_ok=False)
                    logger.info(f"{ome_dir} is created")

                omeconverter(inp_dir=fl, file_pattern=file_pattern, file_extension=".ome.tif", out_dir=ome_dir)

                logger.info(f"Extracting nyxus feature {name}")

                nyxdir = Path(out_dir, name, split_string)
    
                if not nyxdir.exists():
                    nyxdir.mkdir(exist_ok=True, parents=True)

            
                fps = fp.FilePattern(ome_dir, ".*.ome.tif")
                image_path = [f[1][0] for f in fps()][0]
                br = BioReader(image_path)
                if br.physical_size_y[0] is not None:
                    pixels_per_micron = br.physical_size_y[0]
                else:
                    pixels_per_micron = 1.0

                logger.info(f'pixels_per_micro: {pixels_per_micron}')

                nyxfun(intensity_dir=ome_dir, file_pattern=".*.ome.tif", out_dir=nyxdir, pixels_per_micron = pixels_per_micron)

                arrowpath = Path(nyxdir, 'NyxusFeatures.arrow')

                x = vaex.open(arrowpath)
                x['dataset'] =  np.array([name] * len(x))
                x['plate'] =  np.array([fl.name] * len(x))
                x['path'] =  np.array([str(ome_dir)] * len(x))

                finishtime = (time.time() - starttime) / 60
                logger.info(f'total time taken in minutes {finishtime}')
                x.export_feather(nyxdir.joinpath(f"combined_NyxusFeatures.arrow"))



if __name__ == '__main__':
    app()


