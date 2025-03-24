#!/usr/bin/env python3

import csv
from pathlib import Path
import json
from typing import List, Dict, Optional
import typer
from pydantic import BaseModel, ValidationError
import openai

app = typer.Typer(help="CLI tool to process metafiles and generate updated plate files CSV via API")

class PlateFile(BaseModel):
    """Model representing a plate file entry."""
    plate_id: str
    path: str

class ApiResponse(BaseModel):
    """Model for the API response structure."""
    updated_plate_files: List[Dict[str, str]]

def read_plate_csv(csv_file_path: Path) -> List[Dict[str, str]]:
    """
    Read a CSV file containing plate information into a list of dictionaries.

    Args:
        csv_file_path: Path to the CSV file.

    Returns:
        List: List of plate file dictionaries with 'plate_id' and 'path' keys.

    Raises:
        FileNotFoundError: If the CSV file is not found.
        ValueError: If required columns are missing.
    """
    plate_files = []
    try:
        with open(csv_file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            if 'PlateID' not in reader.fieldnames or 'Path' not in reader.fieldnames:
                raise ValueError("CSV file must contain 'PlateID' and 'Path' columns")
            for row in reader:
                plate_files.append({"plate_id": row["PlateID"], "path": row["Path"]})
    except FileNotFoundError:
        typer.echo(f"Error: File not found at {csv_file_path}")
        raise
    except Exception as e:
        typer.echo(f"Error reading CSV: {e}")
        raise
    return plate_files

def get_all_paths(directory: Path) -> List[str]:
    """
    Recursively get all paths in a directory.

    Args:
        directory: Directory path to scan.

    Returns:
        List: List of all paths as strings.

    Raises:
        PermissionError: If access to the directory is denied.
    """
    all_paths = []
    try:
        all_paths.append(str(directory))
        for item in directory.iterdir():
            if item.is_dir():
                all_paths.extend(get_all_paths(item))
    except PermissionError:
        typer.echo(f"Permission denied: {directory}")
        raise
    return all_paths

def process_metafiles(metafiles: List[str]) -> str:
    """
    Process metafiles (CSV and directories) into a text prompt.

    Args:
        metafiles (List[str]): List of metafile paths (CSV files or directories).

    Returns:
        str: Combined text prompt with newline-separated entries.
    """
    plate_files = []
    directory_list = []

    for p in metafiles:
        path = Path(p)
        if ('.csv') in path.name:
            plate_files.extend(read_plate_csv(path))
        else:
            directory_list.append(str(path))

    # Expand directories
    expanded_directory_list = []
    for dir_path in directory_list:
        expanded_directory_list.extend(get_all_paths(Path(dir_path)))

    # Create text prompt
    plate_strings = [f"{plate['plate_id']}: {plate['path']}" for plate in plate_files]
    combined_list = plate_strings + expanded_directory_list
    return "\n".join(combined_list)

def call_api(prompt_file: Path, text_prompt: str) -> str:
    """
    Call the OpenAI API with the given prompt and user content.

    Args:
        prompt_file (Path): Path to the system prompt file.
        text_prompt (str): User content for the API call.

    Returns:
        str: Raw response content from the API.

    Raises:
        Exception: If the API call fails.
    """
    with open(prompt_file, 'r') as f:
        prompt = f.read()

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text_prompt},
    ]

    client = openai.OpenAI(base_url="http://localhost:4000/v1", api_key="sk-1212")
    try:
        typer.echo("Attempting API call...")
        response = client.chat.completions.create(model="gpt4o", messages=messages)
        typer.echo("API call successful")
        return response.choices[0].message.content
    except Exception as e:
        typer.echo(f"API call failed: {e}")
        raise

def write_csv(response_content: str, output_file: Path) -> None:
    """
    Parse API response and write to CSV.

    Args:
        response_content (str): Raw API response content.
        output_file (Path): Path to output CSV file.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
        ValueError: If the expected JSON structure is missing.
        PermissionError: If writing to the file fails.
    """
    # Clean the response
    cleaned_content = response_content.strip()
    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content[7:]
    if cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[:-3]
    cleaned_content = cleaned_content.strip()

    try:
        json_response = json.loads(cleaned_content)
        ApiResponse(**json_response)
    except json.JSONDecodeError as e:
        typer.echo(f"JSON Error: {e}")
        typer.echo(f"Cleaned content: {cleaned_content}")
        raise
    except ValidationError as ve:
        typer.echo(f"Validation Error: {ve}")
        typer.echo(f"Cleaned content: {cleaned_content}")
        raise

    if "updated_plate_files" not in json_response:
        raise ValueError("Response JSON does not contain 'updated_plate_files' key")

    try:
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["PlateID", "Path"])
            if not json_response["updated_plate_files"]:
                typer.echo("Warning: No updated plate files in response")
            for plate in json_response["updated_plate_files"]:
                writer.writerow([plate["plateID"], plate["path"]])
        typer.echo(f"CSV written successfully to {output_file}")
    except PermissionError as pe:
        typer.echo(f"Permission Error: {pe}. Try running with sudo or check directory permissions.")
        raise

@app.command()
def main(
    csv_file: Path = typer.Option(
        ..., "--csvFile", "-c", help="Path of CSV file"
    ),
    dir_path: Path = typer.Option(
        ..., "--dirPath", "-d", help="Path of the raw directory of a  study"
    ),
    prompt_file: Path = typer.Option(
        ..., "--prompt", "-p", help="Path to the system prompt file"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to the output CSV file"
    )
):
    """
    Process metafiles, call an API, and generate an updated plate files CSV.

    Args:
        csv_file: Path to the input CSV file.
        dir_path: Path to the raw data directory for the study.
        prompt_file: Path to the file containing the system prompt.
        output_file: Path where the resulting CSV will be saved.
    """
    metafiles = [csv_file , dir_path]
    if output_file:
        output_file = output_file.joinpath(f"updated_{csv_file.name}")
    else:
        output_file = f"updated_{csv_file.name}"
        
    try:
        text_prompt = process_metafiles(metafiles)
        response_content = call_api(prompt_file, text_prompt)
        write_csv(response_content, output_file)
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()

