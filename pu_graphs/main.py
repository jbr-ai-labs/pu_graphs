from pathlib import Path

import dgl
import torch
from catalyst.engines import DeviceEngine
from catalyst import dl
from catalyst.utils.config import load_config
from torch.utils.data import DataLoader

from pu_graphs.data.datasets import DglGraphDataset
from pu_graphs.data.negative_sampling import UniformStrategy
from pu_graphs.data.utils import get_split
from pu_graphs.modeling.dist_mult import DistMult


def main():
    config = load_config("config.yaml")  # FIXME: replace hardcode

    fb15 = dgl.data.FB15k237Dataset()
    graph = fb15[0]

    datasets = {
        split_key: get_split(graph, split_key)
        for split_key in ("train", "valid", "test")
    }

    loaders = {
        split_key: DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        for split_key, dataset in datasets.items()
    }

    # We assume that each node in present in every graph: train, valid, test
    model = DistMult(
        n_nodes=graph.number_of_nodes(),
        embedding_dim=config["embedding_dim"]
    )
    optimizer = torch.optim.Adam(model.parameters())
    criterion = None # TODO: init
    callbacks = [

    ]  # TODO: init

    runner = dl.SupervisedRunner()
    runner.train(
        engine=dl.DeviceEngine(),
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        callbacks=callbacks,
        verbose=True,
        logdir=Path("./logdir") / config["run_name"],
        num_epochs=config["num_epochs"],
    )

    runner.predict_loader()


if __name__ == '__main__':
    main()
