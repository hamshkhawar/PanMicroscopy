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
        if not original_path or os.path.exists(original_path):
            return original_path  # Return original if it exists or is None

        base_dir = '/projects/PanMicroscopy/data/Idr/raw'
        target_leaf = os.path.basename(original_path)

        # Recursively search for a valid parent directory containing files
        for root, dirs, files in os.walk(base_dir):
            if target_leaf in dirs or target_leaf in os.path.basename(root):
                candidate_path = os.path.join(root, target_leaf if target_leaf in dirs else '')
                if os.path.exists(candidate_path):
                    # Check if this path or any parent contains files
                    current_path = candidate_path
                    while current_path != base_dir:
                        if any(os.path.isfile(os.path.join(current_path, f)) for f in os.listdir(current_path)):
                            return current_path
                        current_path = os.path.dirname(current_path)
                    # If no files found in any parent, return the candidate path
                    return candidate_path
        return original_path  # Fallback to original if not found

    @validator('file_count', pre=True, always=True)
    def count_files_recursively(cls, v, values):
        path = values.get('updated_path', values.get('path'))
        if not path or not os.path.exists(path) or not os.path.isdir(path):
            return 0
        
        total_files = 0
        for _, _, files in os.walk(path):
            total_files += len(files)
        return total_files

    @validator('subdirectories', pre=True, always=True)
    def get_subdirectory_counts(cls, v, values):
        path = values.get('updated_path', values.get('path'))
        if not path or not os.path.exists(path) or not os.path.isdir(path):
            return {}
        
        subdir_counts = {}
        for root, _, files in os.walk(path):
            if files:  # Only include directories with files
                rel_path = os.path.relpath(root, path)
                if rel_path != '.':
                    subdir_counts[rel_path] = len(files)
        return subdir_counts

# Function to process CSV file and write updated CSV without duplicating paths
def process_csv(input_file: str, output_file: str):
    updated_rows = []
    subdirectory_info = {}
    
    # Read the input CSV file
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        seen_paths = set()  # Track unique paths to avoid duplication
        
        for row in reader:
            try:
                # Create Pydantic model instance for each row
                plate = PlateData(plate_id=row['PlateID'], path=row['Path'])
                
                # Skip if path is already processed
                effective_path = plate.updated_path or plate.path
                if effective_path in seen_paths:
                    continue
                seen_paths.add(effective_path)

                # Append processed data as a dict
                updated_rows.append({
                    'PlateID': plate.plate_id,
                    'Path': plate.path,
                    'Exists': plate.exists,
                    'FileCount': plate.file_count,
                    'UpdatedPath': plate.updated_path if plate.updated_path != plate.path else ""
                })
                
                # Store subdirectory information without duplication
                for subdir, count in (plate.subdirectories or {}).items():
                    full_path = os.path.join(effective_path, subdir)
                    if full_path not in subdirectory_info:  # Avoid duplicates
                        subdirectory_info[full_path] = {
                            'PlateID': plate.plate_id,
                            'Subdirectory': subdir,
                            'FullPath': full_path,
                            'FileCount': count
                        }
                
            except Exception as e:
                typer.echo(f"Error processing row {row}: {e}")
    
    # Write the updated data to the main CSV
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['PlateID', 'Path', 'Exists', 'FileCount', 'UpdatedPath']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    
    # Write subdirectory information to a separate CSV
    subdir_output = output_file  # Overwrite the same file for simplicity, adjust if separate file needed
    with open(subdir_output, 'w', newline='') as f:
        fieldnames = ['PlateID', 'Subdirectory', 'FullPath', 'FileCount']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subdirectory_info.values())
    
    typer.echo(f"Processed main CSV saved to {output_file}")
    typer.echo(f"Subdirectory information saved to {subdir_output}")

@app.command()
def main(
    in_file: Path = typer.Option(..., "--inFile", help="Path to the input CSV file"),
    out_dir: Optional[Path] = typer.Option(None, "--outDir", help="Path to the output directory")
):
    """Process a CSV file to validate paths and count files."""
    if out_dir:
        out_file = out_dir.joinpath(f"updated_{in_file.name}")
    else:
        out_file = in_file.parent.joinpath(f"updated_{in_file.name}")

    # Process the CSV file
    process_csv(in_file, out_file)

if __name__ == "__main__":
    app()
