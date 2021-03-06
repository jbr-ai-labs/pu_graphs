import os
import gzip
from dataclasses import dataclass
from io import BytesIO

import dgl
import pandas as pd
import requests
import torch
import numpy as np
from sklearn.utils import shuffle
from dgl.data import DGLDataset


@dataclass
class PolypharmacyDataConstants:
    datapath: str = "data"
    url: str = "http://snap.stanford.edu/biodata/datasets/10017/files/ChChSe-Decagon_polypharmacy.csv.gz"
    download_folder: str = "/downloads"
    txt_dir: str = "/polypharmacy"
    data_file: str = "/data.csv"
    save_dir: str = "/polypharmacy_data"
    graph_file: str = "/dgl_graph.bin"
    info_file: str = "/info.pkl"
    split_ratio: float = 0.1
    min_split_size: int = 50


class PolypharmacyDataset(DGLDataset):
    """
    Polypharmacy dataset in DGL style.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    force_reload: bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information
    """

    def __init__(self,
                 force_reload=False,
                 verbose=False,
                 reversed=False):
        raw_dir = PolypharmacyDataConstants.datapath + PolypharmacyDataConstants.download_folder
        save_dir = PolypharmacyDataConstants.datapath + PolypharmacyDataConstants.save_dir

        self.reversed = reversed
        self.min_split_size = PolypharmacyDataConstants.min_split_size // 2 if reversed else PolypharmacyDataConstants.min_split_size
        self.polypharmacy_path = os.path.abspath(raw_dir) + PolypharmacyDataConstants.txt_dir
        self.path_data = self.polypharmacy_path + PolypharmacyDataConstants.data_file
        self._num_rels = None
        self._num_nodes = None
        self._g = None
        self._train = None
        self._valid = None
        self._test = None
        super(PolypharmacyDataset, self).__init__(name='Polypharmacy',
                                                  url=PolypharmacyDataConstants.url,
                                                  raw_dir=raw_dir,
                                                  save_dir=save_dir,
                                                  force_reload=force_reload,
                                                  verbose=verbose)

    def download(self):
        """
        Download data if it's not presented on a device.
        """

        # download raw data to local disk
        if not os.path.exists(self.polypharmacy_path):
            os.makedirs(self.polypharmacy_path, exist_ok=True)

        if not os.path.exists(self.path_data):
            resp = requests.get(self.url, allow_redirects=True)
            gzipfl = gzip.GzipFile(fileobj=BytesIO(resp.content))

            with open(self.path_data, 'w') as file:
                for row in gzipfl.readlines():
                    file.write(row.decode())

    def _split_data(self, df):
        df_train_list, df_valid_list, df_test_list = [], [], []
        for edge in set(df[1]):
            train_edge, valid_edge, test_edge = self._split_by_edge_type(df[df[1] == edge])
            df_train_list.append(train_edge)
            df_valid_list.append(valid_edge)
            df_test_list.append(test_edge)

        df_train = pd.concat(df_train_list).reset_index(drop=True)
        df_valid = pd.concat(df_valid_list).reset_index(drop=True)
        df_test = pd.concat(df_test_list).reset_index(drop=True)

        return df_train, df_valid, df_test

    def _split_by_edge_type(self, df):
        num_test = max(self.min_split_size,
                       int(np.floor(df.shape[0] * PolypharmacyDataConstants.split_ratio)))
        num_val = num_test
        num_train = df.shape[0] - num_val - num_test
        df = shuffle(df)
        df_train = df[:num_train]
        df_valid = df[num_train:num_train + num_val]
        df_test = df[num_train + num_val:]

        return df_train, df_valid, df_test

    def process(self):
        def _add_reversed_edges(dataframe):
            # Polypharmacy graph is undirected, so we don't need to add new edges type to make reverse edges.
            reversed_df = dataframe.copy()
            reversed_df[0] = dataframe[2].copy()
            reversed_df[2] = dataframe[0].copy()

            return pd.concat([dataframe, reversed_df]).reset_index(drop=True)

        # process raw data to graph
        df = pd.read_csv(self.path_data)
        df.drop(columns=[df.columns[3]], inplace=True)

        # rearrange columns
        tmp_rel = df[df.columns[2]].copy()
        df[df.columns[2]] = df[df.columns[1]].copy()
        df[df.columns[1]] = tmp_rel

        # rename columns
        cols = [0, 1, 2]
        df.columns = cols

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
        self._num_rels = df_enum.shape[0]

        df_train, df_valid, df_test = self._split_data(df)

        if reversed:
            df_train = _add_reversed_edges(df_train)
            df_valid = _add_reversed_edges(df_valid)
            df_test = _add_reversed_edges(df_test)

        # create graph

        graph, data = self.build_knowledge_graph(self._num_nodes, self._num_rels, df_train.values, df_valid.values,
                                                 df_test.values, reverse=self.reversed)
        etype, ntype, train_edge_mask, valid_edge_mask, test_edge_mask, train_mask, val_mask, test_mask = data

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
        graph.ndata['ntype'] = ntype

        # add separation
        graph.edata['train_edge_mask'] = train_edge_mask
        graph.edata['valid_edge_mask'] = valid_edge_mask
        graph.edata['test_edge_mask'] = test_edge_mask
        graph.edata['train_mask'] = train_mask
        graph.edata['val_mask'] = val_mask
        graph.edata['test_mask'] = test_mask

        # add edge data
        graph.edata['etype'] = etype

        self._g = graph

    @staticmethod
    def build_knowledge_graph(num_nodes, num_rels, train, valid, test, reverse=True):
        """
        DGL functions for creation a DGL Homogeneous graph with heterograph info stored as node or edge features
        """
        raw_subg = {}
        raw_subg_eset = {}
        raw_subg_etype = {}

        # there is only one node type
        s_type = "node"
        d_type = "node"

        def add_edge(source, rel, destination, edge_set):
            r_type = str(rel)
            e_type = (s_type, r_type, d_type)
            if raw_subg.get(e_type, None) is None:
                raw_subg[e_type] = ([], [])
                raw_subg_eset[e_type] = []
                raw_subg_etype[e_type] = []
            raw_subg[e_type][0].append(source)
            raw_subg[e_type][1].append(destination)
            raw_subg_eset[e_type].append(edge_set)
            raw_subg_etype[e_type].append(rel)

        for edge in train:
            s, r, d = edge
            assert r < num_rels
            add_edge(s, r, d, 1)  # train set

        for edge in valid:
            s, r, d = edge
            assert r < num_rels
            add_edge(s, r, d, 2)  # valid set

        for edge in test:
            s, r, d = edge
            assert r < num_rels
            add_edge(s, r, d, 3)  # test set

        fg_s = []
        fg_d = []
        fg_etype = []
        fg_settype = []

        for e_type, val in raw_subg.items():
            s, d = val
            s = np.asarray(s)
            d = np.asarray(d)
            etype = raw_subg_etype[e_type]
            etype = np.asarray(etype)
            settype = raw_subg_eset[e_type]
            settype = np.asarray(settype)

            fg_s.append(s)
            fg_d.append(d)
            fg_etype.append(etype)
            fg_settype.append(settype)

        settype = np.concatenate(fg_settype)
        train_edge_mask = torch.tensor(settype == 1)
        valid_edge_mask = torch.tensor(settype == 2)
        test_edge_mask = torch.tensor(settype == 3)

        s = np.concatenate(fg_s)
        d = np.concatenate(fg_d)
        g = dgl.convert.graph((s, d), num_nodes=num_nodes)
        etype = np.concatenate(fg_etype)
        etype = torch.tensor(etype, dtype=torch.int64)
        train_edge_mask = train_edge_mask
        valid_edge_mask = valid_edge_mask
        test_edge_mask = test_edge_mask
        train_mask = train_edge_mask
        valid_mask = valid_edge_mask
        test_mask = test_edge_mask
        ntype = torch.zeros(num_nodes, dtype=torch.int64)

        return g, (etype, ntype, train_edge_mask, valid_edge_mask, test_edge_mask, train_mask, valid_mask, test_mask)

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    def save(self):
        """
        Save processed data to directory `self.save_path`.
        Passed.
        """
        pass

    def load(self):
        """
        Load processed data from directory `self.save_path`.
        Passed.
        """
        pass

    def has_cache(self):
        """
        Check whether there are processed data in `self.save_path`.
        Passed.
        """
        pass
