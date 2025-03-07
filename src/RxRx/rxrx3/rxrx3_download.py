import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
from pathlib import Path
import numpy as np

out_dir = Path('/projects/PanMicroscopy/data/recursionpharma/raw/rxrx3')


num_workers = max([cpu_count() , 2])



def extract_links_and_texts(text):
    soup = BeautifulSoup(text, 'html.parser')
    links_texts = [(a['href'], a.text.split(' ')[0]) for a in soup.find_all('a', href=True)]
    return links_texts

def download_content(url, filename):
    images, compd, fname = filename.split('/')
    outdir = Path(out_dir, images, compd)
    if not Path(outdir).exists():
        Path(outdir).mkdir(exist_ok=True, parents=True)

    output_file_path = Path(outdir, fname)

    try:
        response = requests.get(url, stream=True, verify=True)
        response.raise_for_status()


        with open(str(output_file_path), "wb") as file: 
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Downloaded {filename}")
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")

def main(text_block):
    links_texts = extract_links_and_texts(text_block)


    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        threads = [exe.submit(download_content, link, filename) for link, filename in links_texts]

        for f in tqdm(
            as_completed(threads),
            total=len(threads),
            mininterval=1,
            desc=f"Downloding plate",
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


if __name__ == "__main__":
    # Load text from file linktxt.txt
    with open('/home/abbasih2/projects/PANMicroscopy_scripts/RxRx/rxrx3_links.txt', 'r') as file:
        text_block = file.read()

    main(text_block)