from github import Github, GithubException
from pydantic import BaseModel, conlist
from typing import Optional, Union
import pandas as pd
from io import StringIO, BytesIO
from pathlib import Path
import typer
import logging
import base64
import gzip
import requests

app = typer.Typer()

# Set up logging
logger = logging.getLogger(__name__)


class Submodule(BaseModel):
    path: str
    url: str
    commit_hash: str

class GitHubRepo:
    def __init__(self, access_token: str, repo_name: str, out_dir: Path, root:Path):
        self.github = Github(access_token)
        self.repo = self.github.get_repo(repo_name)
        self.out_dir = out_dir
        self.root = root
        self.access_token = access_token

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
        if content_file.encoding == "base64":
            decoded = base64.b64decode(content_file.content).decode('utf-8')
            logging.info(f"Base64 decoded content size: {len(decoded)} bytes")
            return decoded
        elif content_file.encoding == "none":
            logging.info(f"Content with 'none' encoding, raw size: {len(content_file.content)} bytes")
            logging.info(f"Content preview: {content_file.content[:100]}")
            return content_file.content  
        else:
            raise ValueError(f"Unsupported encoding: {content_file.encoding}")

    def _fetch_raw_content(self, repo_name: str, path: str, ref: str) -> bytes:
        """Fallback method to fetch raw content directly from GitHub."""
        url = f"https://raw.githubusercontent.com/{repo_name}/{ref}/{path}"
        headers = {"Authorization": f"token {self.access_token}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content = response.content
            logging.info(f"Fetched raw content from {url}, size: {len(content)} bytes")
            logging.info(f"Raw content preview: {content[:100]}")
            return content
        else:
            logging.error(f"Failed to fetch raw content from {url}: {response.status_code}")
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
    
    def process_csv_file(self, file_content: bytes, file_path: Path, is_gzipped: bool = False) -> Optional[pd.DataFrame]:
        """Process CSV file content (regular or gzipped) and save to disk."""
        try:
            logging.info(f"Processing {file_path.name}: content size = {len(file_content)} bytes")
            if not file_content:
                logging.warning(f"Skipping {file_path.name}: Empty content")
                return None
                
            if is_gzipped:
                with gzip.open(BytesIO(file_content), 'rt', encoding='utf-8') as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(BytesIO(file_content))
            
            if df.empty:
                logging.warning(f"Skipping {file_path.name}: No data parsed from CSV")
                return None
                
            df.to_csv(file_path, index=False)
            logging.info(f"Saved CSV to {file_path}")
            return df
        except Exception as e:
            logging.error(f"Error processing CSV file {file_path}: {e}")
            return None
        
    def modify_path(self, path):
        if "/uod/idr/filesets/" in path:
            new_path = Path(self.root) / path.split("filesets/")[1]
            return str(Path(new_path).parent)
        else:
            new_path = Path(self.root) / path.split("../", 1)[-1]
            return str(Path(new_path).parent)

    def get_files_in_submodule(self, submodule: Submodule) -> None:
        """Retrieve TSV and CSV files in the submodule at the specified commit."""
        submodule_repo_name = submodule.url.split("github.com/")[-1].replace(".git", "")
        logging.info(f"Accessing submodule repository: {submodule_repo_name}")
        submodule_repo = self.github.get_repo(submodule_repo_name)

        try:
            submodule_commit = submodule_repo.get_commit(submodule.commit_hash)
            submodule_tree = submodule_commit.commit.tree
            logging.info(f"Submodule tree at commit {submodule.commit_hash} retrieved")

            for element in submodule_tree.tree:
                if element.type == "tree":
                    logging.info(f"Scanning directory: {element.path}")
                    dir_contents = submodule_repo.get_contents(element.path, ref=submodule.commit_hash)
                    
                    for subfile in dir_contents:
                        logging.info(f"Examining file: {subfile.path}")
                        # Handle TSV files
                        if subfile.name.endswith(("filePaths.tsv", "plates.tsv")):
                            logging.info(f"Found TSV file: {subfile.name}")
                            tsv_content = submodule_repo.get_contents(subfile.path, ref=submodule.commit_hash)
                            file_content = self._decode_content(tsv_content)
                            if not file_content.strip():
                                logging.warning(f"Skipping {subfile.name}: Empty content")
                                continue
                            file_data = StringIO(file_content)
                            df = pd.read_csv(file_data, sep='\t', header=None)
                            df.rename(columns={df.columns[0]: 'PlateID', df.columns[1]: 'Path'}, inplace=True)
                            df['Path'] = df['Path'].apply(self.modify_path)
                            outpath = self.out_dir.joinpath(f"plates")
                            if not Path(outpath).exists():
                                Path(outpath).mkdir(parents=True, exist_ok=True)
                                
                            plate_path=outpath.joinpath(f"{Path(subfile.name).stem}.csv")
                            df.to_csv(plate_path, index=False)
                        
                        # Handle CSV files (both regular and gzipped)
                        elif subfile.name.endswith(("annotation.csv", "annotation.csv.gz", "annotations.csv")):
                            logging.info(f"Found CSV file: {subfile.name}")
                            file_content_obj = submodule_repo.get_contents(subfile.path, ref=submodule.commit_hash)
                            logging.info(f"Raw content object retrieved for {subfile.name}")
                            file_content = (self._decode_content(file_content_obj).encode('utf-8') 
                                          if not subfile.name.endswith(".gz") 
                                          else file_content_obj.decoded_content)
                            logging.info(f"Content prepared for {subfile.name}, size: {len(file_content)} bytes")
                            if not file_content:
                                logging.warning(f"Content empty after decoding, attempting raw fetch for {subfile.name}")
                                file_content = self._fetch_raw_content(submodule_repo_name, subfile.path, submodule.commit_hash)
                                
                            outpath = self.out_dir.joinpath(f"annotations")
                            if not Path(outpath).exists():
                                Path(outpath).mkdir(parents=True, exist_ok=True)
                            
                            file_path = outpath.joinpath(subfile.name)
                            
                            # Save raw content first
                            with open(file_path, 'wb') as f:
                                f.write(file_content)
                            logging.info(f"Saved raw content to {file_path}")
                            
                            # Process and convert to DataFrame
                            is_gzipped = subfile.name.endswith(".gz")
                            df = self.process_csv_file(file_content, file_path, is_gzipped)
                            if df is not None:
                                logging.info(f"Successfully processed and saved CSV: {subfile.name}")

            return 
            
        except GithubException as e:
            logging.error(f"Error retrieving contents for submodule '{submodule.path}' at commit {submodule.commit_hash}: {e}")
            return None

    def get_files_from_repo(self, name: str) -> None:
        """Retrieve TSV and CSV files from the main repository."""
        contents = self.repo.get_contents(name)

        for content_file in contents:
            if content_file.type == "dir":
                subdir_contents = self.repo.get_contents(content_file.path)
                for subfile in subdir_contents:
                    if subfile.name.endswith("filePaths.tsv") or subfile.name.endswith("plates.tsv"):
                        logging.info(f"Found TSV file: {subfile.name}")
                        tsv_content = self.repo.get_contents(subfile.path)
                        file_content = self._decode_content(tsv_content)
                        if not file_content.strip():
                            logging.warning(f"Skipping {subfile.name}: Empty content")
                            continue
                        file_data = StringIO(file_content)
                        df = pd.read_csv(file_data, sep='\t', header=None)
                        df.rename(columns={df.columns[0]: 'PlateID', df.columns[1]: 'Path'}, inplace=True)
                        df['Path'] = df['Path'].apply(self.modify_path)
                        outpath = self.out_dir.joinpath(f"plates")
                        if not Path(outpath).exists():
                            Path(outpath).mkdir(parents=True, exist_ok=True)
                                
                        plate_path=outpath.joinpath(f"{Path(subfile.name).stem}.csv")
                        df.to_csv(plate_path, index=False)
                    elif subfile.name.endswith(("annotation.csv", "annotation.csv.gz", "annotations.csv")):
                        logging.info(f"Found CSV file: {subfile.name}")
                        file_content_obj = self.repo.get_contents(subfile.path)
                        file_content = (self._decode_content(file_content_obj).encode('utf-8') 
                                      if not subfile.name.endswith(".gz") 
                                      else file_content_obj.decoded_content)
                        if not file_content:
                            logging.warning(f"Content empty after decoding, attempting raw fetch for {subfile.name}")
                            file_content = self._fetch_raw_content(self.repo.full_name, subfile.path, self.repo.default_branch)
                            
                        outpath = self.out_dir.joinpath(f"annotations")
                        if not Path(outpath).exists():
                            Path(outpath).mkdir(parents=True, exist_ok=True)
                            
                        file_path = outpath.joinpath(subfile.name)
                        
                        with open(file_path, 'wb') as f:
                            f.write(file_content)
                        
                        is_gzipped = subfile.name.endswith(".gz")
                        df = self.process_csv_file(file_content, file_path, is_gzipped)
                        if df is not None:
                            logging.info(f"Successfully processed and saved CSV: {subfile.name}")

        return 


# @app.command()
# def main(
#     inp_dir: Optional[Path] = typer.Option(
#         "/Users/abbasih2/Documents/Job/Axle_Work/AI_models/outdir",
#         "--inpDir",
#         help="Path to the input directory",
#         exists=True,
#         resolve_path=True,
#         readable=True,
#         file_okay=False,
#         dir_okay=True,
#     ),
#     name: str = typer.Option(
#         ...,
#         "--name",
#         help="Name of Idr dataset",
#     ),
#     out_dir: Optional[Path] = typer.Option(
#         "/Users/abbasih2/Documents/Job/Axle_Work/AI_models/outdir",
#         "--outDir",
#         help="Output directory",
#         exists=True,
#         resolve_path=True,
#         readable=True,
#         file_okay=False,
#         dir_okay=True,
#     ),
# ):
#     ACCESS_TOKEN = "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"
    
#     github_repo = GitHubRepo(ACCESS_TOKEN, 'IDR/idr-metadata', out_dir, inp_dir)
#     gitmodules_content = github_repo.get_gitmodules_content()

#     if gitmodules_content:
#         submodule = github_repo.find_submodule_in_gitmodules(name, gitmodules_content)
#         if not submodule:
#             github_repo.get_files_from_repo(name)
#         else:
#             logging.info(f"Found submodule '{submodule.path}' with URL: {submodule.url}")
#             submodule.commit_hash = github_repo.get_submodule_commit_hash(name)

#             if submodule.commit_hash:
#                 logging.info(f"Submodule commit hash: {submodule.commit_hash}")
#                 github_repo.get_files_in_submodule(submodule)
#             else:
#                 logging.error(f"Submodule '{name}' not found in the repository tree.")
#     else:
#         logging.info("No content found in .gitmodules.")

# if __name__ == "__main__":
#     app()