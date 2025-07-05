
import gdown
import os
import zipfile

def download_wotr_dataset(file_id, output_dir, output_file):
    """Download the WOTR dataset from Google Drive."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading WOTR dataset to {output_path}...")
    gdown.download(url, output_path, quiet=False)
    print("Download completed!")
    return output_path

def unzip_wotr_dataset(zip_path, extract_dir, extract_folder):
    """Unzip the WOTR dataset to the specified directory."""
    extract_path = os.path.join(extract_dir, extract_folder)
    os.makedirs(extract_path, exist_ok=True)
    print(f"Unzipping {zip_path} to {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Unzipping completed!")