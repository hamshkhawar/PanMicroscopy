from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import filepattern as fp
from tqdm import tqdm 
import os
from bfio import BioReader
import numpy as np
from multiprocessing import cpu_count

from nyxus import Nyxus


# def nyxfun(intensity_dir, file_pattern, outname, out_dir, minI, maxI):
def nyxfun(intensity_dir, file_pattern, outname, out_dir):

    out_name= os.path.join(out_dir, outname)

    nyx = Nyxus(["*ALL*"])

    nyx_params = {
        "neighbor_distance": 5,
        "pixels_per_micron": 1.0,
        "n_feature_calc_threads": 28,
        # 'min_intensity' : minI,
        # 'max_intensity' : maxI
    }

    nyx.set_params(**nyx_params)
    
    nyx.featurize_directory(intensity_dir=str(intensity_dir), 
                            label_dir=None,
                            file_pattern=str(file_pattern),
                            output_type = "arrowipc",
                            output_path = out_name
                            )
            


numworkers = int(cpu_count() / 2)
data_paths = ["val"]
# data_paths = ["train", "val", "test"]
# versions = ["v1.0", "v1.1"]
versions = ["v1.1"]
for v in versions:
    for d in data_paths:
        inp_dir=Path(f'/projects/PanMicroscopy/data/tissueNet/{v}/standard/{d}/intensity')
        out_dir = Path(f'/projects/PanMicroscopy/NyxusFeatures/tissueNet/{v}/standard/{d}')

        if not out_dir.exists():
            out_dir.mkdir(exist_ok=True, parents=True)


        fps = fp.FilePattern(inp_dir, ".*.ome.tif")

        filelist = [f[1][0] for f in fps()]

        with ThreadPoolExecutor(max_workers=50) as executor:
            threads = []

            for inp in filelist:
                outname = Path(inp).name.split('.')[0] + ".arrow"
                threads.append(executor.submit(nyxfun(intensity_dir=inp_dir, 
                file_pattern=Path(inp).name, 
                out_dir=out_dir,
                outname=outname)))

            for f in tqdm(
                    as_completed(threads), desc=f"Processing version:{v} directory: {d}, filename: {inp}", total=len(threads)
                ):
                f.result()


        #     nyxfun(intensity_dir=inp_dir, 
        #                     file_pattern=Path(inp).name, 
        #                     out_dir=out_dir,
        #                     outname=outname)
        #     print(f"Processed:{v} directory: {d}, filename: {inp}")

      

         
          

        # with ThreadPoolExecutor(max_workers=numworkers) as executor:
        #     threads = []

        #     for inp in filelist:
        #         # br = BioReader(inp)
        #         # image = br.read()
        #         # minI = np.min(image)
        #         # maxI = np.max(image)

        #         # br.close()

        #         outname = Path(inp).name.split('.')[0] + ".arrow"

        #         threads.append(executor.submit(nyxfun(intensity_dir=inp_dir, 
        #                     file_pattern=Path(inp).name, 
        #                     out_dir=out_dir,
        #                     outname=outname)))
            
        #     for f in tqdm(
        #             as_completed(threads), desc=f"Processing version:{v} directory: {d}, filename: {inp}", total=len(threads)
        #         ):
        #         f.result()

            # try:
            #     nyxfun(intensity_dir=inp_dir, 
            #             file_pattern=Path(inp).name, 
            #             out_dir=out_dir,
            #             outname=outname,
            #             # minI=minI,
            #             # maxI =maxI
            #             )
            #     print(f"processed: {inp}")
                 
            # except Exception as e:
            #     print(f"Error: {e} - for file {inp}.")


  

           

