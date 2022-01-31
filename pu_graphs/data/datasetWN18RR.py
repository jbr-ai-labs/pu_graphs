from distutils.command.build import build
import os
import zipfile
from dataclasses import dataclass
from io import BytesIO

import dgl
import pandas as pd
import requests
import torch
import numpy as np
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
                 reversed=False):
        raw_dir = WN18RRDataConstants.datapath + WN18RRDataConstants.download_folder
        save_dir = WN18RRDataConstants.datapath + WN18RRDataConstants.save_dir

        self.reversed = reversed
        self.wn18rr_path = os.path.abspath(raw_dir)
        self.path_train = self.wn18rr_path + WN18RRDataConstants.txt_dir + WN18RRDataConstants.train_file
        self.path_valid = self.wn18rr_path + WN18RRDataConstants.txt_dir + WN18RRDataConstants.valid_file
        self.path_test = self.wn18rr_path + WN18RRDataConstants.txt_dir + WN18RRDataConstants.test_file
        super(WN18RRDataset, self).__init__(name='WN18RR',
                                            url=WN18RRDataConstants.url,
                                            raw_dir=raw_dir,
                                            save_dir=save_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)

    def download(self):
        # download raw data to local disk
        if not os.path.exists(self.wn18rr_path):
            os.makedirs(self.wn18rr_path, exist_ok=True)
    
        if not (
                os.path.exists(self.path_train)
                and os.path.exists(self.path_valid)
                and os.path.exists(self.path_test)):
            resp = requests.get(self.url, allow_redirects=True)
            zipfl = zipfile.ZipFile(BytesIO(resp.content))
            zipfl.extractall(self.wn18rr_path)


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
        self._num_rels = df_enum.shape[0]

        df_train = df.iloc[:len(df_train)]
        df_valid = df.iloc[len(df_train) : len(df_train) + len(df_valid)]
        df_test = df.iloc[len(df_train) + len(df_valid) : len(df_train) + len(df_valid) + len(df_test)]

        # create graph

        graph, data = self.build_knowledge_graph(self._num_nodes, self._num_rels, df_train.values, df_valid.values, df_test.values, reverse=self.reversed)
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
        print(graph.device)
        graph.ndata['ntype'] = ntype

        # add separation
        graph.edata['train_edge_mask'] = train_edge_mask
        graph.edata['valid_edge_mask'] = valid_edge_mask
        graph.edata['test_edge_mask'] = test_edge_mask
        graph.edata['train_mask'] = train_mask
        graph.edata['val_mask'] = val_mask
        graph.edata['test_mask'] = test_mask

        # add edge data
        orig = list(set(df[1]))
        df_enum = pd.DataFrame({'original': orig, 'enum': list(range(len(orig)))})
        df_enum = df_enum.set_index('original')
        graph.edata['etype'] = etype
        print(graph.edata['etype'].numpy())

        self._g = graph

    @staticmethod
    def build_knowledge_graph(num_nodes, num_rels, train, valid, test, reverse=True):
        """
        DGL funciton for creation a DGL Homogeneous graph with heterograph info stored as node or edge features
        """
        src = []
        rel = []
        dst = []
        raw_subg = {}
        raw_subg_eset = {}
        raw_subg_etype = {}
        raw_reverse_sugb = {}
        raw_reverse_subg_eset = {}
        raw_reverse_subg_etype = {}

        # here there is noly one node type
        s_type = "node"
        d_type = "node"

        def add_edge(s, r, d, reverse, edge_set):
            r_type = str(r)
            e_type = (s_type, r_type, d_type)
            if raw_subg.get(e_type, None) is None:
                raw_subg[e_type] = ([], [])
                raw_subg_eset[e_type] = []
                raw_subg_etype[e_type] = []
            raw_subg[e_type][0].append(s)
            raw_subg[e_type][1].append(d)
            raw_subg_eset[e_type].append(edge_set)
            raw_subg_etype[e_type].append(r)

            if reverse is True:
                r_type = str(r + num_rels)
                re_type = (d_type, r_type, s_type)
                if raw_reverse_sugb.get(re_type, None) is None:
                    raw_reverse_sugb[re_type] = ([], [])
                    raw_reverse_subg_etype[re_type] = []
                    raw_reverse_subg_eset[re_type] = []
                raw_reverse_sugb[re_type][0].append(d)
                raw_reverse_sugb[re_type][1].append(s)
                raw_reverse_subg_eset[re_type].append(edge_set)
                raw_reverse_subg_etype[re_type].append(r + num_rels)

        for edge in train:
            s, r, d = edge
            assert r < num_rels
            add_edge(s, r, d, reverse, 1) # train set

        for edge in valid:
            s, r, d = edge
            assert r < num_rels
            add_edge(s, r, d, reverse, 2) # valid set

        for edge in test:
            s, r, d = edge
            assert r < num_rels
            add_edge(s, r, d, reverse, 3) # test set

        subg = []
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
        if reverse is True:
            settype = np.concatenate([settype, np.full((settype.shape[0]), 0)])
        train_edge_mask = torch.tensor(settype == 1)
        valid_edge_mask = torch.tensor(settype == 2)
        test_edge_mask = torch.tensor(settype == 3)

        for e_type, val in raw_reverse_sugb.items():
            s, d = val
            s = np.asarray(s)
            d = np.asarray(d)
            etype = raw_reverse_subg_etype[e_type]
            etype = np.asarray(etype)
            settype = raw_reverse_subg_eset[e_type]
            settype = np.asarray(settype)

            fg_s.append(s)
            fg_d.append(d)
            fg_etype.append(etype)
            fg_settype.append(settype)

        s = np.concatenate(fg_s)
        d = np.concatenate(fg_d)
        g = dgl.convert.graph((s, d), num_nodes=num_nodes)
        etype = np.concatenate(fg_etype)
        settype = np.concatenate(fg_settype)
        etype = torch.tensor(etype, dtype=torch.int64)
        train_edge_mask = train_edge_mask
        valid_edge_mask = valid_edge_mask
        test_edge_mask = test_edge_mask
        train_mask = torch.tensor(settype == 1) if reverse is True else train_edge_mask
        valid_mask = torch.tensor(settype == 2) if reverse is True else valid_edge_mask
        test_mask = torch.tensor(settype == 3) if reverse is True else test_edge_mask
        ntype = torch.zeros(num_nodes, dtype = torch.int64)

        return g, (etype, ntype, train_edge_mask, valid_edge_mask, test_edge_mask, train_mask, valid_mask, test_mask)

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    def save(self):
        return 
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
        return 
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
        return 
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, WN18RRDataConstants.graph_file)
        info_path = os.path.join(self.save_path, WN18RRDataConstants.info_file)
        return os.path.exists(graph_path) and os.path.exists(info_path)
