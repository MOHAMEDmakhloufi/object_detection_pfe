import gdown
import os
import zipfile

class DataDownloader:
    def __init__(self, file_id, output_dir, output_file, extract_dir, extract_folder):
        """
        Initialize the DataDownloader.

        Args:
            file_id (str): The Google Drive file ID of the dataset.
            output_dir (str): The directory where the zip file will be downloaded.
            output_file (str): The name of the zip file to be saved.
            extract_dir (str): The base directory where the dataset will be extracted.
            extract_folder (str): The subfolder within extract_dir where the dataset will be extracted.
        """
        self.file_id = file_id
        self.output_dir = output_dir
        self.output_file = output_file
        self.extract_dir = extract_dir
        self.extract_folder = extract_folder
        self.zip_path = None

    def download(self):
        """
        Download the WOTR dataset from Google Drive.
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.zip_path = os.path.join(self.output_dir, self.output_file)
            url = f"https://drive.google.com/uc?id={self.file_id}"
            print(f"Downloading WOTR dataset to {self.zip_path}...")
            gdown.download(url, self.zip_path, quiet=False)
            print("Download completed!")
        except Exception as e:
            print(f"Error during download: {e}")
            self.zip_path = None

    def unzip(self):
        """
        Unzip the downloaded WOTR dataset.
        """
        if self.zip_path is None:
            print("No zip file to unzip. Please run download first.")
            return
        try:
            extract_path = os.path.join(self.extract_dir, self.extract_folder)
            os.makedirs(extract_path, exist_ok=True)
            print(f"Unzipping {self.zip_path} to {extract_path}...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("Unzipping completed!")
        except Exception as e:
            print(f"Error during unzipping: {e}")

    def download_and_unzip(self):
        """
        Download and unzip the WOTR dataset in one step.
        """
        self.download()
        if self.zip_path:
            self.unzip()