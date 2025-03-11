import os
import csv
from pydantic import BaseModel, validator
from typing import Optional, Dict
import typer
from pathlib import Path

app = typer.Typer()

# Pydantic model for each row in the CSV
class PlateData(BaseModel):
    plate_id: str
    path: str
    exists: Optional[bool] = None
    file_count: Optional[int] = None
    updated_path: Optional[str] = None
    subdirectories: Optional[Dict[str, int]] = None

    @validator('exists', pre=True, always=True)
    def check_path_exists(cls, v, values):
        path = values.get('path')
        if path:
            return os.path.exists(path)
        return False

    @validator('updated_path', pre=True, always=True)
    def find_updated_path(cls, v, values):
        original_path = values.get('path')
        if original_path and not os.path.exists(original_path):
            base_dir = '/projects/PanMicroscopy/data/Idr/raw' 
            target_leaf = os.path.basename(original_path) 
            for root, dirs, _ in os.walk(base_dir):
                if target_leaf in dirs or target_leaf in root:
                    updated = os.path.join(root, target_leaf)
                    if os.path.exists(updated):
                        return updated
            return original_path  # Fallback to original if not found
        return original_path

    @validator('file_count', pre=True, always=True)
    def count_files_recursively(cls, v, values):
        # Determine which path to use (updated_path if it exists, otherwise original path)
        path = values.get('updated_path', values.get('path'))
        
        if not path or not os.path.exists(path) or not os.path.isdir(path):
            return 0
        
        # Count files recursively
        total_files = 0
        for _, _, files in os.walk(path):
            total_files += len(files)
        
        return total_files

    @validator('subdirectories', pre=True, always=True)
    def get_subdirectory_counts(cls, v, values):
        # Determine which path to use (updated_path if it exists, otherwise original path)
        path = values.get('updated_path', values.get('path'))
        
        if not path or not os.path.exists(path) or not os.path.isdir(path):
            return {}
        
        # Track subdirectory counts
        subdir_counts = {}
        for root, _, files in os.walk(path):
            if len(files) > 0:  # Only include directories with files
                rel_path = os.path.relpath(root, path)
                if rel_path != '.':  # Skip the root directory itself
                    subdir_counts[rel_path] = len(files)
        
        return subdir_counts

# Function to read CSV file, process it, and write updated CSV
def process_csv(input_file: str, output_file: str):
    updated_rows = []
    subdirectory_info = []
    
    # Read the input CSV file
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Create Pydantic model instance for each row
                plate = PlateData(plate_id=row['PlateID'], path=row['Path'])
                
                # Append processed data as a dict
                updated_rows.append({
                    'PlateID': plate.plate_id,
                    'Path': plate.path,
                    'Exists': plate.exists,
                    'FileCount': plate.file_count,
                    'UpdatedPath': plate.updated_path if plate.updated_path != plate.path else ""
                })
                
                # Store subdirectory information
                for subdir, count in (plate.subdirectories or {}).items():
                    if count > 0:  # Only include subdirectories with files
                        full_path = os.path.join(plate.updated_path or plate.path, subdir)
                        subdirectory_info.append({
                            'PlateID': plate.plate_id,
                            'Subdirectory': subdir,
                            'FullPath': full_path,
                            'FileCount': count
                        })
                
            except Exception as e:
                typer.echo(f"Error processing row {row}: {e}")
    
    # Write the updated data to the main CSV
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['PlateID', 'Path', 'Exists', 'FileCount', 'UpdatedPath']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    
    # Write subdirectory information to a separate CSV
    subdir_output = os.path.splitext(output_file)[0] + ".csv"
    with open(subdir_output, 'w', newline='') as f:
        fieldnames = ['PlateID', 'Subdirectory', 'FullPath', 'FileCount']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subdirectory_info)
    
    typer.echo(f"Processed main CSV saved to {output_file}")
    typer.echo(f"Subdirectory information saved to {subdir_output}")

@app.command()
def main(
    in_file: Path = typer.Option(..., "--inFile", help="Path to the input CSV file"),
    out_dir: Optional[Path] = typer.Option(None, "--outDir", help="Path to the output CSV file")
):
    """Process a CSV file to validate paths and count files."""
    # process_csv(in_file, out_file)

    if out_dir:
        out_file = out_dir.joinpath(f"updated_{in_file.name}")
    else:
        out_file = in_file.parent.joinpath(f"updated_{in_file.name}")

      # Process the CSV file
    process_csv(in_file, out_file)



if __name__ == "__main__":
    app()

