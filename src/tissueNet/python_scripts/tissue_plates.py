import time
from typing import Optional
from pathlib import Path
import logging
import re
import typer



app = typer.Typer()

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("Idr dataset")
logger.setLevel(logging.INFO)

def find_directories_with_images(root_dir, extensions=['.jpg', '.png', '.jpeg', '.tif', '.bmp', '.ics', ".DIB", '.ome.tif', '.ome.zarr']):
        """
        Recursively find images with specific extensions, excluding paths 
        that contain 'Ground_Truth', 'Masks', or 'masks', and return the highest parent folder names 
        (removes subdirectories when parent is already added).
        
        Args:
            root_dir (str or Path): The root directory to search for images.
            extensions (list): List of valid image file extensions to search for.
        
        Returns:
            set: Set of unique folder paths with the highest-level parents only.
        """
        # Convert extensions to lowercase to ensure case-insensitive matching
        extensions = [ext.lower() for ext in extensions]
        
        # List to store image paths and folder names
        unique_folders = set()

        # Recursively search for image files
        for file_path in Path(root_dir).rglob('*'):
            # Check if the file is an image with the right extension
            if file_path.suffix.lower() in extensions:
                # Check if the path contains 'Ground_Truth', 'Masks', or 'masks'
                if not re.search(r'raw|label', str(file_path)):
                    folder_path = file_path.parent  # Get the folder path
                    
                # Check if any existing folder in the set is a subdirectory of the new folder
                if not any(str(existing_folder).startswith(str(folder_path)) for existing_folder in unique_folders):
                    # Remove any subdirectories of the current folder
                    unique_folders = {existing_folder for existing_folder in unique_folders if not str(folder_path).startswith(str(existing_folder))}
                    unique_folders.add(folder_path)  # Add the current folder path

        return unique_folders



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


    plates_paths = find_directories_with_images(root_dir=inp_dir)

    filename = Path.cwd().joinpath(f"plates.txt")
    with open(str(filename), 'w') as f:
        for p in sorted(plates_paths):
            f.write(str(p)+"\n")





if __name__ == '__main__':
    app()