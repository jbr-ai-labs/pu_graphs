import typing as ty
from io import BytesIO

import requests
import dgl
import torch
from .constants import DataConstants
import zipfile
import os


def get_split(graph: dgl.DGLGraph, split_key: str) -> dgl.DGLGraph:
    split_mask = graph.edata[f"{split_key}_edge_mask"]
    split_edges_index = torch.nonzero(split_mask, as_tuple=False).squeeze()

    split_graph = graph.edge_subgraph(split_edges_index, preserve_nodes=True)
    split_graph.edata["etype"] = graph.edata["etype"][split_edges_index]

    return split_graph

def load_wn18rr():
    wn18rr_path = os.path.abspath(DataConstants.datapath + DataConstants.wn18rr_download_folder)
    if not os.path.exists(wn18rr_path):
        os.makedirs(wn18rr_path,exist_ok=True)
    wn18rr_path_train =  wn18rr_path + DataConstants.wn18rr_txt_dir + DataConstants.wn18rr_train
    wn18rr_path_valid = wn18rr_path + DataConstants.wn18rr_txt_dir + DataConstants.wn18rr_valid
    wn18rr_path_test = wn18rr_path + DataConstants.wn18rr_txt_dir + DataConstants.wn18rr_test
    if not (
            os.path.exists(wn18rr_path_train)
            and os.path.exists(wn18rr_path_valid)
            and os.path.exists(wn18rr_path_test)):
        resp = requests.get(DataConstants.wn18rr_url, allow_redirects=True)
        zipfl = zipfile.ZipFile(BytesIO(resp.content))
        zipfl.extractall(wn18rr_path)



if __name__ == '__main__':
    load_wn18rr()