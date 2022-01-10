from dataclasses import dataclass

@dataclass
class DataConstants:
    datapath: str = "data"
    wn18rr_url: str = "https://data.deepai.org/WN18RR.zip"
    wn18rr_download_folder: str = "/downloads"
    wn18rr_txt_dir: str = "/WN18RR/text"
    wn18rr_train: str = "/train.txt"
    wn18rr_valid: str = "/valid.txt"
    wn18rr_test: str = "/test.txt"
