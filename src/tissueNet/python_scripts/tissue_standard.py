from bfio import BioWriter
import pathlib
from tqdm import tqdm
import numpy as np
import os
import re
import time
from typing import Optional
import multiprocessing
from multiprocessing import cpu_count
from functools import partial
import sys


global FILE_EXT
FILE_EXT = os.environ.get("POLUS_EXT", ".ome.tif")

DATASET = "TissueNet"


def makedirectory(root: str, version: str) -> None:
    """Make directories for downloading raw and preprocessed data"""
    for sub in ["standard", "raw"]:
        dirpath = os.path.join(root, DATASET, version, sub)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            f"{root}/{DATASET}/{version}/{sub} directory created"
        else:
            pass
    return


def save_omezarr(image: np.ndarray, out_file: pathlib.Path) -> None:
    """Writing images in omezarr format
    Args:
        image (np.ndarray): Input image array
        out_file (pathlib.Path): Path to save omezarr image
    """
    with BioWriter(file_path=out_file) as bw:
        bw.X = image.shape[0]
        bw.Y = image.shape[1]
        bw.dtype = image.dtype
        bw[:] = image


def parsing_tissuenet_data(
    npzfilepath: os.path,
    version: str,
    outDir: os.path,
    fileExtension: Optional[str] = ".ome.tif",
    normalize: bool = False,
):
    """Parsing npz files into intensity and label images
    Args:
        npzfilepath (pathlib.Path): Directory path containing downloaded npz files
        version (str): Select either of the two supported TissueNet data versions (v1.0 and v1.1)
        outDir (pathlib.Path): Path to output directory
        fileExtension (str): Type of data conversion. Default format is .ome.tif but can also generate '.ome.zarr' image files
    """
    npzfilename = pathlib.Path(npzfilepath).name
    assert npzfilename.endswith("npz"), 'No file found with ".npz" file extension'
    name = re.split("_|\.", npzfilename)[3]
    vers = re.split("_", npzfilename)[1]

    assert (
        vers == version
    ), f"Provided {version} is not compatible with the npz file {vers}!! Please check it again"

    outfile_intensity = pathlib.Path(outDir, name).joinpath("intensity")
    outfile_label = pathlib.Path(outDir, name).joinpath("label")
    # Create standard subdirectory

    FILE_EXT = FILE_EXT if fileExtension is None else fileExtension

    if not os.path.exists(outfile_intensity):
        outfile_intensity.mkdir(exist_ok=True, parents=True)
    if not os.path.exists(outfile_label):
        outfile_label.mkdir(exist_ok=True, parents=True)
    print("Loading: " + npzfilename)

    data = np.load(npzfilepath, allow_pickle=True)
    if vers == "v1.0":
        X, y, tissue_list, platform_list = (
            data["X"],
            data["y"],
            data["tissue_list"],
            data["platform_list"],
        )
        # sanity check
        num_examples = len(tissue_list)
        assert len(tissue_list) == len(platform_list)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == len(tissue_list)
        tissuemap = {
            "breast": 0,
            "gi": 1,
            "immune": 2,
            "lung": 3,
            "pancreas": 4,
            "skin": 5,
        }
        platformmap = {
            "codex": 0,
            "cycif": 1,
            "imc": 2,
            "mibi": 3,
            "mxif": 4,
            "vectra": 5,
        }
        tissuelist = [tissuemap[v] for v in tissue_list]
        platformlist = [platformmap[v] for v in platform_list]
        for ex, tissue, platform in tqdm(
            zip(range(num_examples + 1), tissuelist, platformlist),
            desc="Extracting images",
            unit="files",
            total=len(tissue_list),
        ):
            image_ch0 = X[ex, :, :, 0].squeeze()
            label_ch0 = y[ex, :, :, 1].squeeze()  
            image_ch1 = X[ex, :, :, 1].squeeze() 
            label_ch1 = y[ex, :, :, 0].squeeze()

            file_outname_ch0 = "p{0}_y{1}_r{2}_c0{3}".format(
                platform, tissue, ex, FILE_EXT
            )
            file_outname_ch1 = "p{0}_y{1}_r{2}_c1{3}".format(
                platform, tissue, ex, FILE_EXT
            )
            
            save_omezarr(
                image_ch0, out_file=pathlib.Path(outfile_intensity, file_outname_ch0)
            )
            save_omezarr(
                label_ch0, out_file=pathlib.Path(outfile_label, file_outname_ch0)
            )
            save_omezarr(
                image_ch1, out_file=pathlib.Path(outfile_intensity, file_outname_ch1)
            )
            save_omezarr(
                label_ch1, out_file=pathlib.Path(outfile_label, file_outname_ch1)
            )

    elif vers == "v1.1":
        if name == "test":
            meta = [idx[5] for idx in data["meta"][4:]]
        else:
            meta = [idx[5] for idx in data["meta"][1:]]

        X, y, tissue_list = (data["X"], data["y"], meta)
        # sanity check
        num_examples = len(tissue_list)
        assert len(tissue_list) == X.shape[0]
        assert X.shape[0] == y.shape[0]

        tissuemap = {
            "Breast": 0,
            "Colon": 1,
            "Lymph Node": 2,
            "Lung": 3,
            "Pancreas": 4,
            "Epidermis": 5,
            "Esophagus": 6,
            "Spleen": 7,
            "Tonsil": 8,
            "lymph node metastasis": 9,
        }

        tissuelist = [tissuemap[v] for v in tissue_list]

        for ex, tissue in tqdm(
            zip(range(num_examples + 1), tissuelist),
            desc="Extracting images",
            unit="files",
            total=len(tissue_list),
        ):

            if normalize:
                image_ch0 = X[ex, :, :, 0].squeeze()
                image_ch0= (image_ch0 - np.min(image_ch0)) / (np.max(image_ch0) - np.min(image_ch0))
                image_ch1 = X[ex, :, :, 1].squeeze()
                image_ch1= (image_ch1 - np.min(image_ch1)) / (np.max(image_ch1) - np.min(image_ch1))
            else:
                image_ch0 = X[ex, :, :, 0].squeeze()
                image_ch1 = X[ex, :, :, 1].squeeze()
            
            label_ch0 = y[ex, :, :, 1].squeeze()
            label_ch1 = y[ex, :, :, 0].squeeze()

            file_outname_ch0 = "y{0}_r{1}_c0{2}".format(tissue, ex, FILE_EXT)
            file_outname_ch1 = "y{0}_r{1}_c1{2}".format(tissue, ex, FILE_EXT)
            save_omezarr(
                image_ch0, out_file=pathlib.Path(outfile_intensity, file_outname_ch0)
            )
            save_omezarr(
                label_ch0, out_file=pathlib.Path(outfile_label, file_outname_ch0)
            )
            save_omezarr(
                image_ch1, out_file=pathlib.Path(outfile_intensity, file_outname_ch1)
            )
            save_omezarr(
                label_ch1, out_file=pathlib.Path(outfile_label, file_outname_ch1)
            )

    return


def tissue_data(root: str, version: str, fileExtension: str, normalize: bool):
    starttime = time.time()
    FILE_EXT = FILE_EXT if fileExtension is None else fileExtension

    print(f"Starting extracting and preparing datasets!!!")

    inpDir = pathlib.Path(root).joinpath("raw")
    outDir = pathlib.Path(root).joinpath("standard")

    npzpath = [
        os.path.join(path, file)
        for path, _, files in os.walk(inpDir)
        for file in files
        if file.endswith(".npz")
    ]

    if len(npzpath) != 3:
        raise ValueError(
            "Incorrect number of compressed files in a directory!!Please check the directory again"
        )

    versions = [f.split("/")[-1].split("_")[-2] for f in npzpath]

    if not all(versions):
        assert (
            version == versions[0]
        ), f"TissueNet data {version} is not compatible with raw npz files! Please check raw data"

    if version == "v1.0":

        if sum([os.stat(f).st_size for f in npzpath]) != sum(
            [596338244, 299654094, 2657531397]
        ):
            raise ValueError(
                "Files are not downloaded completely!!Please download them again"
            )
    elif version == "v1.1":
        if sum([os.stat(f).st_size for f in npzpath]) != sum(
            [371328939, 2873378914, 1220043531]
        ):
            raise ValueError(
                "Files are not downloaded completely!!Please download them again"
            )

    num_workers = max(multiprocessing.cpu_count() // 2, 2)

    with multiprocessing.Pool(processes=num_workers) as executor:

        executor.map(
            partial(
                parsing_tissuenet_data,
                version=version,
                outDir=outDir,
                fileExtension=fileExtension,
                normalize=normalize,
            ),
            npzpath,
        )

        executor.close()
        executor.join()
    endtime = round((time.time() - starttime) / 60, 3)
    print(f"Time taken to parse tissueNet dataset: {endtime} minutes!!!")


def main():
    root = str(sys.argv[1])
    version = str(sys.argv[2])
    fileExtension = str(sys.argv[3])
    normalize = sys.argv[4]
    

if __name__ == "__main__":
    main()
