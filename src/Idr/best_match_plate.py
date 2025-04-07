import os
import csv
import typer
from pathlib import Path
from typing import Optional, List

app = typer.Typer(help="Update plate ID paths in a CSV file by searching a directory structure")

def find_plate_dir(root_dir: Path, plate_id: str, case_sensitive: bool = False) -> Optional[Path]:
    """
    Search recursively through root_dir for a directory or file containing the plate_id.
    Prioritizes exact matches and checks if directories contain files.
    
    Args:
        root_dir (Path): The root directory to start searching from
        plate_id (str): The plate ID to search for
        case_sensitive (bool): Whether matching should be case-sensitive
        
    Returns:
        Path: Full path to the best matching directory containing the plate_id, or None if not found
    """
    typer.echo(f"Searching for {plate_id} in {root_dir}...")
    
    # Handle case sensitivity
    search_id = plate_id if case_sensitive else plate_id.lower()
    best_match = None
    best_match_score = 0  # Higher score = better match
    
    for root, dirs, files in os.walk(root_dir):
        current_path = Path(root)
        base_dir = current_path.name
        check_dir = base_dir if case_sensitive else base_dir.lower()
        
        # Score 3: Exact directory name match (highest priority)
        if search_id == check_dir:
            full_path = current_path
            has_files = bool(files)  # Check if directory contains files
            typer.echo(f"Found exact directory match: {full_path} (contains files: {has_files})")
            return full_path
        
        # Score 2: Plate ID is contained in directory name
        if search_id in check_dir and best_match_score < 2:
            has_files = bool(files)
            best_match = current_path
            best_match_score = 2
            typer.echo(f"Found partial directory match: {best_match} (contains files: {has_files})")
            
        # Check subdirectories
        for d in dirs:
            check_d = d if case_sensitive else d.lower()
            subdir_path = current_path / d
            # Score 3: Exact subdirectory match
            if search_id == check_d:
                has_files = any(subdir_path.iterdir())  # Check if subdirectory contains any files
                typer.echo(f"Found exact subdirectory match: {subdir_path} (contains files: {has_files})")
                return subdir_path
            # Score 1: Partial subdirectory match
            elif search_id in check_d and best_match_score < 1:
                has_files = any(subdir_path.iterdir())
                best_match = subdir_path
                best_match_score = 1
                typer.echo(f"Found partial subdirectory match: {best_match} (contains files: {has_files})")
        
        # Check files
        for file in files:
            check_file = file if case_sensitive else file.lower()
            # Score 2: Exact file match
            if search_id == Path(check_file).stem and best_match_score < 2:
                best_match = current_path
                best_match_score = 2
                typer.echo(f"Found exact file match: {current_path / file}")
            # Score 1: Partial file match
            elif search_id in check_file and best_match_score < 1:
                best_match = current_path
                best_match_score = 1
                typer.echo(f"Found partial file match: {current_path / file}")
    
    if best_match:
        has_files = any(best_match.iterdir())  # Final check for files in best match
        typer.echo(f"Returning best match: {best_match} (score: {best_match_score}, contains files: {has_files})")
        return best_match
        
    typer.echo(f"No matches found for {plate_id}")
    return None

def has_files_with_format(directory: Path, file_formats: List[str]) -> List[Path]:
    """
    Check if directory contains files with specified formats and return them.
    
    Args:
        directory (Path): Directory to check
        file_formats (List[str]): List of file extensions to look for (without the dot)
        
    Returns:
        List[Path]: List of matching files
    """
    if not directory.exists() or not directory.is_dir():
        return []
        
    matching_files = []
    for item in directory.iterdir():
        if item.is_file() and any(item.name.lower().endswith(f".{fmt.lower()}") for fmt in file_formats):
            matching_files.append(item)
    
    return matching_files

@app.command()
def update_paths(
    csv_file: Path = typer.Argument(..., help="Path to the input CSV file with plate IDs and paths"),
    root_dir: Path = typer.Argument(..., help="Root directory to search for plate IDs"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to the output CSV file"),
    file_formats: List[str] = typer.Option([], "--format", "-f", help="File formats to look for (e.g., jpg, tif, png)"),
    case_sensitive: bool = typer.Option(False, "--case-sensitive", "-c", help="Make the search case-sensitive")
):
    """
    Update plate ID paths in a CSV file by recursively searching a directory structure.
    Updates the path when a plate ID match is found and reports on specified file formats.
    """
    if output is None:
        output = csv_file.with_stem(f"{csv_file.stem}_updated")
    
    if file_formats:
        typer.echo(f"Will report on files with formats: {', '.join(file_formats)}")
    
    updated_rows = []
    match_stats = {"exact": 0, "partial": 0, "not_found": 0, "with_matching_files": 0, "without_matching_files": 0}
    
    # Read the input CSV
    with csv_file.open('r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Assuming first row is header
        updated_rows.append(header)
        
        # Process each row
        for row in reader:
            if len(row) >= 2:  # Ensure the row has at least 2 columns
                plate_id, old_path_str = row[0], row[1]
                old_path = Path(old_path_str) if old_path_str else None
                
                # Try old path first if it exists
                search_paths = []
                if old_path and old_path.exists():
                    search_paths.append(old_path)
                search_paths.append(root_dir)
                
                new_path = None
                for path in search_paths:
                    new_path = find_plate_dir(path, plate_id, case_sensitive)
                    if new_path:
                        break
                
                if new_path:
                    row[1] = str(new_path)  # Convert Path to string for CSV
                    
                    # Check for files with specified formats if formats were provided
                    matching_files = []
                    if file_formats:
                        matching_files = has_files_with_format(new_path, file_formats)
                    
                    final_component = new_path.name.lower() if not case_sensitive else new_path.name
                    plate_id_compare = plate_id.lower() if not case_sensitive else plate_id
                    final_component_compare = final_component.lower() if not case_sensitive else final_component
                    
                    if plate_id_compare == final_component_compare:
                        match_stats["exact"] += 1
                        match_type = "exact"
                    else:
                        match_stats["partial"] += 1
                        match_type = "partial"
                    
                    if file_formats:
                        if matching_files:
                            match_stats["with_matching_files"] += 1
                            typer.echo(f"✅ Found {match_type} match for {plate_id} at: {new_path}")
                            typer.echo(f"   Has {len(matching_files)} matching files with formats: {', '.join(file_formats)}")
                            for file in matching_files[:5]:  # Show up to 5 matching files
                                typer.echo(f"   - {file.name}")
                            if len(matching_files) > 5:
                                typer.echo(f"   - ... and {len(matching_files) - 5} more")
                        else:
                            match_stats["without_matching_files"] += 1
                            typer.echo(f"⚠️ Found {match_type} match for {plate_id} at: {new_path}")
                            typer.echo(f"   No files with formats: {', '.join(file_formats)}")
                    else:
                        typer.echo(f"✅ Found {match_type} match for {plate_id} at: {new_path}")
                else:
                    typer.echo(f"❌ Could not find {plate_id}, keeping original path: {old_path_str}")
                    match_stats["not_found"] += 1
                    row[1] = old_path_str
                
                updated_rows.append(row)
            else:
                typer.echo(f"⚠️ Skipping invalid row: {row}")
                updated_rows.append(row)
    
    # Write the updated data to the output CSV
    with output.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(updated_rows)
    
    # Print summary
    typer.echo("\n" + "="*50)
    typer.echo(f"Summary of results:")
    typer.echo(f"  Exact matches found: {match_stats['exact']}")
    typer.echo(f"  Partial matches found: {match_stats['partial']}")
    typer.echo(f"  Plate IDs not found: {match_stats['not_found']}")
    
    if file_formats:
        typer.echo(f"  Directories with {', '.join(file_formats)} files: {match_stats['with_matching_files']}")
        typer.echo(f"  Directories without {', '.join(file_formats)} files: {match_stats['without_matching_files']}")
    
    typer.echo("="*50)
    typer.echo(f"\nUpdated CSV saved to: {output}")

if __name__ == "__main__":
    app()