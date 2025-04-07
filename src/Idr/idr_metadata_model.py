"""
A script to fetch and process TSV/CSV files from a GitHub repository, including submodules.
Can process a specific dataset or all datasets in the repository.
Now combines multiple plates.tsv files into a single CSV for each study.
"""

from github import Github, GithubException
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
from io import StringIO, BytesIO
from pathlib import Path
import typer
import logging
import base64
import gzip
import requests
import os

app = typer.Typer()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")

class Submodule(BaseModel):
    path: str
    url: str
    commit_hash: str

class GitHubRepo:
    def __init__(self, access_token: str, repo_name: str, out_dir: Path, root: Path):
        self.github = Github(access_token)
        self.repo = self.github.get_repo(repo_name)
        self.out_dir = out_dir
        self.root = root
        self.access_token = access_token
        # Dictionary to store dataframes for each study
        self.study_plates_data: Dict[str, List[pd.DataFrame]] = {}

    def get_gitmodules_content(self) -> Optional[str]:
        """Fetch the contents of the .gitmodules file."""
        try:
            gitmodules_file = self.repo.get_contents(".gitmodules")
            content = self._decode_content(gitmodules_file)
            logging.info(f".gitmodules content retrieved, size: {len(content)} bytes")
            return content
        except GithubException as e:
            logging.error(f".gitmodules file not found: {e}")
            return None

    def _decode_content(self, content_file) -> str:
        """Decode content based on its encoding."""
        logging.info(f"Decoding content for {content_file.path}, encoding: {content_file.encoding}")
        try:
            if content_file.encoding == "base64":
                decoded_bytes = base64.b64decode(content_file.content)
                decoded_str = decoded_bytes.decode('utf-8', errors='replace')
                if not decoded_str.strip():
                    logging.warning(f"Empty content after base64 decoding for {content_file.path}")
                logging.info(f"Base64 decoded content size: {len(decoded_str)} bytes")
                return decoded_str
            elif content_file.encoding == "none":
                logging.info(f"Content with 'none' encoding, raw size: {len(content_file.content)} bytes")
                logging.info(f"Content preview: {content_file.content[:100]}")
                return content_file.content.decode('utf-8', errors='replace')
            else:
                raise ValueError(f"Unsupported encoding: {content_file.encoding}")
        except Exception as e:
            logging.error(f"Decoding failed for {content_file.path}: {e}")
            return ""

    def _fetch_raw_content(self, repo_name: str, path: str, ref: str) -> bytes:
        """Fallback method to fetch raw content directly from GitHub."""
        # Use the GitHub raw URL format
        url = f"https://raw.githubusercontent.com/{repo_name}/{ref}/{path}"
        logging.info(f"Fetching raw content from: {url}")
        
        # Add proper headers for authentication
        headers = {
            "Authorization": f"token {self.access_token}",
            "Accept": "application/vnd.github.v3.raw"
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for non-200 status codes
            
            content = response.content
            logging.info(f"Successfully fetched raw content from {url}, size: {len(content)} bytes")
            if len(content) > 0:
                logging.info(f"Raw content preview: {content[:100]}")
            else:
                logging.warning(f"Empty content received from {url}")
            
            return content
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch raw content from {url}: {e}")
            return b''

    def find_submodule_in_gitmodules(self, target_submodule: str, gitmodules_content: str) -> Optional[Submodule]:
        """Parse .gitmodules content to find the specified submodule."""
        lines = gitmodules_content.splitlines()
        for i, line in enumerate(lines):
            if f'path = {target_submodule}' in line:
                submodule_url_line = lines[i + 1] if i + 1 < len(lines) and "url" in lines[i + 1] else None
                if submodule_url_line:
                    submodule_url = submodule_url_line.split(" = ")[1].strip()
                    return Submodule(path=target_submodule, url=submodule_url, commit_hash="")
        logging.info(f"Submodule '{target_submodule}' not found in .gitmodules.")
        return None

    def get_all_submodules(self, gitmodules_content: str) -> List[Submodule]:
        """Parse .gitmodules content to find all submodules."""
        submodules = []
        lines = gitmodules_content.splitlines()
        i = 0
        
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("[submodule") and i + 2 < len(lines):
                path_line = lines[i + 1]
                url_line = lines[i + 2]
                
                if "path = " in path_line and "url = " in url_line:
                    path = path_line.split("path = ")[1].strip()
                    url = url_line.split("url = ")[1].strip()
                    
                    submodule = Submodule(path=path, url=url, commit_hash="")
                    submodules.append(submodule)
                    
                    logging.info(f"Found submodule: {path} with URL: {url}")
                
                i += 3  # Skip to next potential submodule block
            else:
                i += 1
                
        logging.info(f"Found {len(submodules)} submodules in .gitmodules")
        return submodules

    def get_submodule_commit_hash(self, target_submodule: str) -> Optional[str]:
        """Get the commit hash of the specified submodule from the main repository."""
        main_commit = self.repo.get_branch(self.repo.default_branch).commit
        tree = main_commit.commit.tree

        for element in tree.tree:
            if element.type == "commit" and element.path == target_submodule:
                logging.info(f"Found commit hash for submodule '{target_submodule}': {element.sha}")
                return element.sha
        logging.info(f"Commit hash for submodule '{target_submodule}' not found.")
        return None

    def get_root_dirs(self) -> List[str]:
        """Get all top-level directories in the repo that could be study folders."""
        try:
            contents = self.repo.get_contents("")
            dirs = []
            
            for item in contents:
                if item.type == "dir" and not item.path.startswith("."):
                    # Skip hidden directories and common non-study folders
                    if item.path not in ["scripts", "docs", "utils", "templates", "config", "tests"]:
                        dirs.append(item.path)
                        logging.info(f"Found directory: {item.path}")
            
            return dirs
        except GithubException as e:
            logging.error(f"Error retrieving root directories: {e}")
            return []

    def process_csv_file(self, file_content: bytes, file_path: Path, is_gzipped: bool = False) -> Optional[pd.DataFrame]:
        """Process CSV file content (regular or gzipped) and save to disk."""
        try:
            logging.info(f"Processing {file_path.name}: content size = {len(file_content)} bytes")
            if not file_content or len(file_content) == 0:
                logging.warning(f"Skipping {file_path.name}: Empty content")
                return None

            # First, save the raw bytes to file
            out_dir = file_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(file_content)
            logging.info(f"Saved raw file to {file_path}")

            # Then try to process as DataFrame
            try:
                if is_gzipped:
                    with gzip.open(BytesIO(file_content), 'rt', encoding='utf-8', errors='replace') as f:
                        # Try to detect delimiter
                        sample = f.read(1024)
                        f.seek(0)
                        
                        if '\t' in sample:
                            df = pd.read_csv(f, sep='\t')
                        else:
                            df = pd.read_csv(f)
                else:
                    # Try to detect delimiter
                    sample = file_content[:1024].decode('utf-8', errors='replace')
                    
                    if '\t' in sample:
                        df = pd.read_csv(BytesIO(file_content), sep='\t', encoding='utf-8', errors='replace')
                    else:
                        df = pd.read_csv(BytesIO(file_content), encoding='utf-8', errors='replace')

                if df.empty:
                    logging.warning(f"No data parsed from {file_path.name}")
                    return None

                # Save as CSV (regardless of original format)
                csv_path = file_path.with_suffix('.csv') if file_path.suffix != '.csv' else file_path
                df.to_csv(csv_path, index=False)
                logging.info(f"Successfully parsed and saved DataFrame to {csv_path}, shape: {df.shape}")
                return df
                
            except Exception as e:
                logging.error(f"Error reading {file_path.name} as DataFrame: {e}")
                logging.info(f"Raw file was saved to {file_path}, but couldn't be parsed as DataFrame")
                return None
                
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            return None

    def modify_path(self, path):
        if "/uod/idr/filesets/" in path:
            new_path = Path(self.root) / path.split("filesets/")[1]
            if Path(new_path).suffix:
                return str(Path(new_path).parent)
            return str(Path(new_path))
                
        elif "../" in path:
            new_path = Path(self.root) / path.split("../", 1)[-1] if "../" in path else path
            if "screen" in Path(new_path).suffix:
                return str(Path(new_path).with_suffix(""))
            if not "screen" in Path(new_path).suffix:
                return str(Path(new_path).parent)
            return str(Path(new_path))
        else:
            new_path = Path(self.root) / path
            if Path(new_path).suffix:
                return str(Path(new_path).parent)
            return str(Path(new_path))
            
    def save_combined_plates(self):
        """Save combined plates data for each study."""
        for study_name, dfs in self.study_plates_data.items():
            if not dfs or len(dfs) == 0:
                logging.warning(f"No plate data found for study {study_name}")
                continue
                
            logging.info(f"Combining {len(dfs)} plate files for study {study_name}")
            
            try:
                # Concatenate all dataframes for this study
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Remove duplicates if any
                before_dedup_count = len(combined_df)
                combined_df = combined_df.drop_duplicates()
                after_dedup_count = len(combined_df)
                
                if before_dedup_count > after_dedup_count:
                    logging.info(f"Removed {before_dedup_count - after_dedup_count} duplicate rows")
                
                # Create output directory
                combined_dir = self.out_dir.joinpath("combined_plates")
                combined_dir.mkdir(parents=True, exist_ok=True)
                
                # Save combined file
                combined_path = combined_dir.joinpath(f"{study_name}_combined_plates.csv")
                combined_df.to_csv(combined_path, index=False)
                logging.info(f"Successfully saved combined plates data to {combined_path}, total rows: {len(combined_df)}")
            except Exception as e:
                logging.error(f"Error combining plates data for study {study_name}: {e}")

    def store_plates_df(self, study_name: str, df: pd.DataFrame):
        """Store a plates dataframe for later combination."""
        if study_name not in self.study_plates_data:
            self.study_plates_data[study_name] = []
        
        self.study_plates_data[study_name].append(df)
        logging.info(f"Stored plates data for study {study_name}, now have {len(self.study_plates_data[study_name])} files")

    def get_files_in_submodule(self, submodule: Submodule) -> None:
        """Retrieve TSV and CSV files in the submodule at the specified commit."""
        submodule_repo_name = submodule.url.split("github.com/")[-1].replace(".git", "")
        logging.info(f"Accessing submodule repository: {submodule_repo_name}")
        
        try:
            submodule_repo = self.github.get_repo(submodule_repo_name)
            submodule_commit = submodule_repo.get_commit(submodule.commit_hash)
            submodule_tree = submodule_commit.commit.tree
            logging.info(f"Submodule tree at commit {submodule.commit_hash} retrieved")

            def process_tree_recursive(tree, path_prefix=""):
                for element in tree.tree:
                    current_path = f"{path_prefix}/{element.path}" if path_prefix else element.path
                    
                    if element.type == "tree":
                        logging.info(f"Scanning directory: {current_path}")
                        try:
                            subtree = submodule_repo.get_git_tree(element.sha)
                            process_tree_recursive(subtree, current_path)
                        except GithubException as e:
                            logging.error(f"Error retrieving subtree {current_path}: {e}")
                    
                    elif element.type == "blob":
                        # Handle TSV files
                        if element.path.endswith(("filePaths.tsv", "plates.tsv")):
                            logging.info(f"Found TSV file: {current_path}")
                            
                            # Get file content directly using raw URL
                            file_content = self._fetch_raw_content(
                                submodule_repo_name, 
                                current_path, 
                                submodule.commit_hash
                            )
                            
                            if not file_content:
                                logging.warning(f"Empty content for {current_path}, skipping")
                                continue
                                
                            try:
                                # Process TSV
                                file_data = StringIO(file_content.decode('utf-8', errors='replace'))
                                df = pd.read_csv(file_data, sep='\t', header=None)
                                if len(df.columns) >= 2:  # Ensure we have at least 2 columns
                                    df.rename(columns={df.columns[0]: 'PlateID', df.columns[1]: 'Path'}, inplace=True)
                                    df['Path'] = df['Path'].apply(self.modify_path)
                                    
                                    # Create study-specific output directory
                                    outpath = self.out_dir.joinpath("plates", submodule.path)
                                    outpath.mkdir(parents=True, exist_ok=True)
                                    
                                    plate_path = outpath.joinpath(f"{Path(element.path).stem}.csv")
                                    df.to_csv(plate_path, index=False)
                                    logging.info(f"Successfully saved plate data to {plate_path}")
                                    
                                    # Store for later combination
                                    self.store_plates_df(submodule.path, df)
                                else:
                                    logging.warning(f"TSV file {current_path} has insufficient columns: {df.columns}")
                            except Exception as e:
                                logging.error(f"Error processing TSV file {current_path}: {e}")

                        # Handle CSV files (both regular and gzipped)
                        elif element.path.endswith(("annotation.csv", "annotation.csv.gz", "annotations.csv")):
                            logging.info(f"Found CSV file: {current_path}")
                            
                            # Always use raw content fetch for CSV files to avoid encoding issues
                            file_content = self._fetch_raw_content(
                                submodule_repo_name, 
                                current_path, 
                                submodule.commit_hash
                            )
                            
                            if not file_content or len(file_content) == 0:
                                logging.warning(f"Empty content for {current_path}, skipping")
                                continue
                                
                            is_gzipped = element.path.endswith(".gz")
                            
                            # Create study-specific output directory
                            outpath = self.out_dir.joinpath("annotations", submodule.path)
                            outpath.mkdir(parents=True, exist_ok=True)
                            
                            file_path = outpath.joinpath(Path(element.path).name)
                            df = self.process_csv_file(file_content, file_path, is_gzipped)
                            
                            if df is not None:
                                logging.info(f"Successfully processed and saved CSV: {element.path}")

            # Process the root tree recursively
            process_tree_recursive(submodule_tree)
            return

        except GithubException as e:
            logging.error(f"Error retrieving submodule '{submodule.path}' at commit {submodule.commit_hash}: {e}")
            return None

    def get_files_from_repo(self, name: str) -> None:
        """Retrieve TSV and CSV files from the main repository."""
        try:
            contents = self.repo.get_contents(name)
            
            def process_contents_recursive(contents, path_prefix=""):
                if not isinstance(contents, list):
                    contents = [contents]
                    
                for content_file in contents:
                    current_path = f"{path_prefix}/{content_file.path}" if path_prefix else content_file.path
                    
                    if content_file.type == "dir":
                        logging.info(f"Scanning directory: {current_path}")
                        try:
                            subdir_contents = self.repo.get_contents(content_file.path)
                            process_contents_recursive(subdir_contents, "")
                        except GithubException as e:
                            logging.error(f"Error retrieving contents for {current_path}: {e}")
                    
                    elif content_file.type == "file":
                        # Handle TSV files
                        if content_file.name.endswith(("filePaths.tsv", "plates.tsv")):
                            logging.info(f"Found TSV file: {current_path}")
                            
                            # Get file content directly using raw URL
                            file_content = self._fetch_raw_content(
                                self.repo.full_name, 
                                content_file.path, 
                                self.repo.default_branch
                            )
                            
                            if not file_content:
                                file_content = self._decode_content(content_file).encode('utf-8')
                                
                            if not file_content:
                                logging.warning(f"Empty content for {current_path}, skipping")
                                continue
                                
                            try:
                                # Process TSV
                                file_data = StringIO(file_content.decode('utf-8', errors='replace'))
                                df = pd.read_csv(file_data, sep='\t', header=None)
                                if len(df.columns) >= 2:  # Ensure we have at least 2 columns
                                    df.rename(columns={df.columns[0]: 'PlateID', df.columns[1]: 'Path'}, inplace=True)
                                    df['Path'] = df['Path'].apply(self.modify_path)
                                    
                                    # Create study-specific output directory
                                    outpath = self.out_dir.joinpath("plates", name)
                                    outpath.mkdir(parents=True, exist_ok=True)
                                    
                                    plate_path = outpath.joinpath(f"{Path(content_file.name).stem}.csv")
                                    df.to_csv(plate_path, index=False)
                                    logging.info(f"Successfully saved plate data to {plate_path}")
                                    
                                    # Store for later combination
                                    self.store_plates_df(name, df)
                                else:
                                    logging.warning(f"TSV file {current_path} has insufficient columns: {df.columns}")
                            except Exception as e:
                                logging.error(f"Error processing TSV file {current_path}: {e}")

                        # Handle CSV files (both regular and gzipped)
                        elif content_file.name.endswith(("annotation.csv", "annotation.csv.gz", "annotations.csv")):
                            logging.info(f"Found CSV file: {current_path}")
                            
                            # Always use raw content fetch for CSV files to avoid encoding issues
                            file_content = self._fetch_raw_content(
                                self.repo.full_name, 
                                content_file.path, 
                                self.repo.default_branch
                            )
                            
                            if not file_content or len(file_content) == 0:
                                # Try fallback method
                                try:
                                    file_content = self._decode_content(content_file).encode('utf-8')
                                except Exception as e:
                                    logging.error(f"Fallback decode failed for {current_path}: {e}")
                                
                            if not file_content or len(file_content) == 0:
                                logging.warning(f"Empty content for {current_path} after all attempts, skipping")
                                continue
                                
                            is_gzipped = content_file.name.endswith(".gz")
                            
                            # Create study-specific output directory
                            outpath = self.out_dir.joinpath("annotations", name)
                            outpath.mkdir(parents=True, exist_ok=True)
                            
                            file_path = outpath.joinpath(content_file.name)
                            df = self.process_csv_file(file_content, file_path, is_gzipped)
                            
                            if df is not None:
                                logging.info(f"Successfully processed and saved CSV: {content_file.name}")
            
            # Process contents recursively
            process_contents_recursive(contents)
            return
            
        except GithubException as e:
            logging.error(f"Error retrieving contents for path '{name}': {e}")

    def process_all_studies(self) -> None:
        """Process all studies in the repository."""
        logging.info("Processing all studies in the repository")
        
        # Get all submodules
        gitmodules_content = self.get_gitmodules_content()
        if gitmodules_content:
            submodules = self.get_all_submodules(gitmodules_content)
            logging.info(f"Found {len(submodules)} submodules")
            
            for submodule in submodules:
                logging.info(f"Processing submodule: {submodule.path}")
                submodule.commit_hash = self.get_submodule_commit_hash(submodule.path)
                
                if submodule.commit_hash:
                    logging.info(f"Submodule '{submodule.path}' commit hash: {submodule.commit_hash}")
                    self.get_files_in_submodule(submodule)
                else:
                    logging.warning(f"Couldn't find commit hash for submodule '{submodule.path}', skipping")
        
        # Also check top-level directories
        root_dirs = self.get_root_dirs()
        for dir_name in root_dirs:
            logging.info(f"Processing directory: {dir_name}")
            self.get_files_from_repo(dir_name)
        
        # After processing all studies, combine plates files for each study
        self.save_combined_plates()

@app.command()
def main(
    root: Path = typer.Option(
        ...,
        "--root",
        help="Path to the input directory",
        exists=True,
        resolve_path=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Name of Idr dataset",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        help="Output directory",
        exists=True,
        resolve_path=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
    ),
    all: bool = typer.Option(
        False,
        "--all",
        help="If True, process all studies in the repository",
    ),
):
    # Ensure we have a valid access token
    if not ACCESS_TOKEN:
        logging.error("No ACCESS_TOKEN found in environment variables")
        raise ValueError("GitHub ACCESS_TOKEN environment variable must be set")
    
    # Create output directories if they don't exist
    annotations_dir = out_dir.joinpath("annotations")
    plates_dir = out_dir.joinpath("plates")
    combined_dir = out_dir.joinpath("combined_plates")
    annotations_dir.mkdir(parents=True, exist_ok=True)
    plates_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting GitHub repository processing")
    logging.info(f"Output directory: {out_dir}")
    logging.info(f"Root directory: {root}")
    logging.info(f"Process all studies: {all}")
    
    github_repo = GitHubRepo(ACCESS_TOKEN, 'IDR/idr-metadata', out_dir, root)

    if (name is None and not all) or (name is not None and all):
        typer.echo("Error: You must provide either --name or --all (but not both).")
        raise typer.Exit(code=1)
    
    if all:
        logging.info("Processing all studies in the repository")
        github_repo.process_all_studies()
    else:
        logging.info(f"Processing specific study: {name}")
        gitmodules_content = github_repo.get_gitmodules_content()

        if gitmodules_content:
            submodule = github_repo.find_submodule_in_gitmodules(name, gitmodules_content)
            if not submodule:
                logging.info(f"No submodule found for '{name}', processing as a directory in the main repository")
                github_repo.get_files_from_repo(name)
            else:
                logging.info(f"Found submodule '{submodule.path}' with URL: {submodule.url}")
                submodule.commit_hash = github_repo.get_submodule_commit_hash(name)

                if submodule.commit_hash:
                    logging.info(f"Submodule commit hash: {submodule.commit_hash}")
                    github_repo.get_files_in_submodule(submodule)
                else:
                    logging.error(f"Submodule '{name}' commit hash not found, check if the submodule exists")
        else:
            logging.info("No .gitmodules file found, processing as a directory in the main repository")
            github_repo.get_files_from_repo(name)
        
        # After processing the specific study, combine its plates files
        github_repo.save_combined_plates()
    
    logging.info("Processing completed")

if __name__ == "__main__":
    app()