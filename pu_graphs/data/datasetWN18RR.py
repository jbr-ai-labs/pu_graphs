import os
import zipfile
from dataclasses import dataclass
from io import BytesIO

import dgl
import pandas as pd
import requests
import torch
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.data.utils import save_info, load_info


@dataclass
class WN18RRDataConstants:
    datapath: str = "data"
    url: str = "https://data.deepai.org/WN18RR.zip"
    download_folder: str = "/downloads"
    txt_dir: str = "/WN18RR/text"
    train_file: str = "/train.txt"
    valid_file: str = "/valid.txt"
    test_file: str = "/test.txt"
    save_dir: str = "/wn18rr_data"
    graph_file: str = "/dgl_graph.bin"
    info_file: str = "/info.pkl"


class WN18RRDataset(DGLDataset):
    """
    WN18RR dataset in DGL style.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    def __init__(self,
                 force_reload=False,
                 verbose=False,
                 reversed=True):
        super(WN18RRDataset, self).__init__(name='WN18RR',
                                            url=WN18RRDataConstants.url,
                                            raw_dir=WN18RRDataConstants.download_folder,
                                            save_dir=WN18RRDataConstants.save_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)
        self.reversed = reversed

    def download(self):
        # download raw data to local disk
        wn18rr_path = os.path.abspath(self.raw_dir)
        if not os.path.exists(wn18rr_path):
            os.makedirs(wn18rr_path, exist_ok=True)
        self.path_train = wn18rr_path + WN18RRDataConstants.txt_dir + WN18RRDataConstants.train_file
        self.path_valid = wn18rr_path + WN18RRDataConstants.txt_dir + WN18RRDataConstants.valid_file
        self.path_test = wn18rr_path + WN18RRDataConstants.txt_dir + WN18RRDataConstants.test_file
        if not (
                os.path.exists(self.path_train)
                and os.path.exists(self.path_valid)
                and os.path.exists(self.path_test)):
            resp = requests.get(self.url, allow_redirects=True)
            zipfl = zipfile.ZipFile(BytesIO(resp.content))
            zipfl.extractall(wn18rr_path)


    def process(self):
        # process raw data to graph
        df_train = pd.read_table(self.path_train, sep='\t', header=None)
        df_valid = pd.read_table(self.path_valid, sep='\t', header=None)
        df_test = pd.read_table(self.path_test, sep='\t', header=None)

        df = pd.concat([df_train, df_valid, df_test])

        # create graph

        # enumerate nodes
        orig = list(set(df[0]).union(set(df[2])))
        df_enum = pd.DataFrame({'original': orig, 'enum': list(range(len(orig)))})
        df_enum = df_enum.set_index('original')
        df[0] = df_enum.loc[df[0]].enum.values
        df[2] = df_enum.loc[df[2]].enum.values
        self._num_nodes = len(orig)

        # enumerate edges
        orig = list(set(df[1]))
        df_enum = pd.DataFrame({'original': orig, 'enum': list(range(len(orig)))})
        df_enum = df_enum.set_index('original')
        df[1] = df_enum.loc[df[1]].enum.values
        self._num_rels = df_enum.shape[1]

        # create graph
        graph_dict = dict()
        for rel in self._num_rels:
            df_rel = df[df[1] == rel]
            graph_dict[('word', rel, 'word')] = (torch.tensor(list(df_rel[0])), torch.tensor(list(df_rel[2])))
            if self.reversed:
                # add reversed relations
                graph_dict[('word', rel+self._num_rels, 'word')] = (torch.tensor(list(df_rel[2])), torch.tensor(list(df_rel[0])))
        graph = dgl.heterograph(graph_dict)

        # for compatability
        self._train = df_train
        self._valid = df_valid
        self._test = df_test

        if self.verbose:
            print("# entities: {}".format(self._num_nodes))
            print("# relations: {}".format(self._num_rels))
            print("# training edges: {}".format(self._train.shape[0]))
            print("# validation edges: {}".format(self._valid.shape[0]))
            print("# testing edges: {}".format(self._test.shape[0]))

        # add node information
        graph.ndata['ntype'] = torch.tensor([0]*self._num_nodes)

        # add separation
        num_edges = 2*len(df) if self.reversed else len(df)
        train_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)

        n_train = len(self._train)
        n_train_with_reversed = 2 * len(self._train) if self.reversed else len(self._train)
        n_val = len(self._valid)
        n_val_with_reversed = 2 * len(self._valid) if self.reversed else len(self._valid)
        n_test = len(self._test)
        train_edge_mask[:n_train] = True
        val_edge_mask[n_train:n_train_with_reversed + n_val] = True
        test_edge_mask[n_train_with_reversed + n_val_with_reversed:n_train_with_reversed + n_val_with_reversed + n_test] = True
        train_mask[:n_train_with_reversed] = True
        val_mask[n_train_with_reversed:n_train_with_reversed + n_val_with_reversed] = True
        test_mask[n_train_with_reversed + n_val_with_reversed:] = True
        graph.edata['train_edge_mask'] = train_edge_mask
        graph.edata['val_edge_mask'] = val_edge_mask
        graph.edata['test_edge_mask'] = test_edge_mask
        graph.edata['train_mask'] = train_mask
        graph.edata['val_mask'] = val_mask
        graph.edata['test_mask'] = test_mask

        # add edge data
        orig = list(set(df[1]))
        df_enum = pd.DataFrame({'original': orig, 'enum': list(range(len(orig)))})
        df_enum = df_enum.set_index('original')
        graph.edata['etype'] = df_enum.loc[df[1]].enum.values

        self._g = graph

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    def save(self):
        # save processed data to directory `self.save_path`
        # save graphs and labels
        graph_path = os.path.join(self.save_path, WN18RRDataConstants.graph_file)
        graph_labels = {"glabel": torch.tensor(list(range(self._num_rels)))}
        save_graphs(graph_path, self._g, graph_labels)
        # save other information in python dict
        info_path = os.path.join(self.save_path, 'info.pkl')
        save_info(info_path, {'df_train': self._train,
                              'df_val': self._valid,
                              'df_test': self._test,
                              'num_nodes': self._num_nodes,
                              'num_rels': self._num_rels})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, WN18RRDataConstants.graph_file)
        self._g, _ = load_graphs(graph_path)
        info_path = os.path.join(self.save_path, WN18RRDataConstants.info_file)
        info = load_info(info_path)['num_classes']
        self._train = info['df_train']
        self._valid = info['df_val']
        self._test = info['df_test']
        self._num_nodes = info['num_nodes']
        self._num_rels = info['num_rels']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, WN18RRDataConstants.graph_file)
        info_path = os.path.join(self.save_path, WN18RRDataConstants.info_file)
        return os.path.exists(graph_path) and os.path.exists(info_path)
