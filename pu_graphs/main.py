import os
from pathlib import Path

import dgl
import hydra_slayer
import numpy as np
# import sparse
import torch
from catalyst import dl
from catalyst.utils import set_global_seed
from catalyst.utils.config import load_config
from torch.utils.data import DataLoader

from pu_graphs.data.datasets import DglGraphDataset
from pu_graphs.data.datasetWN18RR import WN18RRDataset
from pu_graphs.data.negative_sampling import UniformStrategy
from pu_graphs.debug_utils import DebugDataset
from pu_graphs.evaluation.callback import EvaluationCallback
from pu_graphs.evaluation.evaluation import MRRLinkPredictionMetric, \
    AccuracyLinkPredictionMetric, AdjustedMeanRankIndex, FilteredLinkPredictionMetric
from pu_graphs.modeling.dist_mult import DistMult
from pu_graphs.modeling.loss import UnbiasedPULoss, logistic_loss


def evaluation_callback(graphs, loaders, eval_loader_key: str, is_debug: bool):
    full_graph = graphs["train"]
    eval_graph = graphs[eval_loader_key]

    number_of_nodes = full_graph.number_of_nodes()
    number_of_relations = eval_graph.edata["etype"].max().item() + 1

    head_idx, tail_idx = full_graph.edges()
    head_idx = head_idx.numpy()
    tail_idx = tail_idx.numpy()
    relation_idx = full_graph.edata["etype"].numpy()

    coordinates = (relation_idx, head_idx, tail_idx)
    values = np.ones_like(head_idx)

    full_adj_mat = sparse.COO((values, coordinates), shape=[number_of_relations, number_of_nodes, number_of_nodes])

    metrics_builders = {
        "mrr": lambda suf: MRRLinkPredictionMetric(topk_args=[number_of_nodes], suffix=suf),
        "acc": lambda suf: AccuracyLinkPredictionMetric(topk_args=[1, 3, 5, 10, 20], suffix=suf),
        "amri": lambda suf: AdjustedMeanRankIndex(topk_args=[full_graph.number_of_nodes()], suffix=suf)
    }

    metrics = {}
    for k, v in metrics_builders.items():
        metrics[k] = v("")
        metrics[f"{k}_filtered"] = FilteredLinkPredictionMetric(metric=v("_filtered"), full_adj_mat=full_adj_mat)

    return EvaluationCallback(
        graph=eval_graph,
        metrics=metrics,
        loader=loaders[eval_loader_key],
        loader_key=eval_loader_key,
        is_debug=is_debug
    )


def main():
    config = hydra_slayer.get_from_params(
        **load_config("config.yaml")  # FIXME: replace hardcode
    )
    is_debug = config["is_debug"]

    if is_debug:
        os.environ["WANDB_MODE"] = "dryrun"

    set_global_seed(config["seed"])

    if config['dataset'] == 'FB15k237':
        dataset = dgl.data.FB15k237Dataset()
    elif config['dataset'] == 'WN18RR':
        dataset = WN18RRDataset()
    full_graph = dataset[0]

    graphs = {
        k: get_split(full_graph, k)
        for k in ("train", "valid", "test")
    }
    datasets = {
        split_key: DglGraphDataset(
            graph=graph,
            strategy=UniformStrategy(graph)
        )
        for split_key, graph in graphs.items()
    }

    if is_debug:
        datasets = {k: DebugDataset(v, n_examples=100) for k, v in datasets.items()}

    loaders = {
        split_key: DataLoader(
            dataset,
            batch_size=config["batch_size"] if split_key == "train" else config["eval_batch_size"],
            shuffle=True
        )
        for split_key, dataset in datasets.items()
    }
    train_and_valid_loaders = {
        k: v for k, v in loaders.items()
        if k != "test"
    }

    # We assume that each node in present in every graph: train, valid, test
    model = DistMult(
        n_nodes=full_graph.number_of_nodes(),
        n_relations=graphs["train"].edata["etype"].max().item() + 1,
        embedding_dim=config["embedding_dim"]
    )
    optimizer = config["optimizer"](model.parameters())
    criterion = UnbiasedPULoss(logistic_loss, pi=0.5)  # FIXME: estimate pi somehow

    def transform_as_pos_neg(batch):
        n_positives = batch["head_indices"].size(0)
        n_unlabeled = batch["neg_head_indices"].size(0)
        new_fields = {
            "head_indices": torch.cat([batch["head_indices"], batch["neg_head_indices"]]),
            "tail_indices": torch.cat([batch["tail_indices"], batch["neg_tail_indices"]]),
            "relation_indices": batch["relation_indices"].expand([2, -1]).flatten(),
            "labels": torch.cat([torch.ones(n_positives), torch.zeros(n_unlabeled)]),
        }
        batch.update(new_fields)
        return batch

    logdir = Path("./logdir") / config["run_name"]
    callbacks = [
        dl.BatchTransformCallback(
            transform=transform_as_pos_neg,
            scope="on_batch_start"
        ),  # TODO: maybe move negative sampling to this callback too
        dl.EarlyStoppingCallback(
            patience=config["patience"],
            loader_key="valid",
            metric_key="loss",  # TODO: replace with MRR,
            minimize=True
        ),
        dl.CheckpointCallback(
            logdir=logdir.joinpath("checkpoints"),
            loader_key="valid",
            metric_key="loss",
            minimize=True,
            load_on_stage_end="best"
        ),
        evaluation_callback(
            graphs=graphs,
            loaders=loaders,
            eval_loader_key="test",
            is_debug=is_debug
        )
    ]

    loggers = {
        "wandb": dl.WandbLogger(project="pu_graphs", name=config["run_name"])
    }

    runner = dl.SupervisedRunner(
        input_key=["head_indices", "tail_indices", "relation_indices"],
        output_key="logits",
        target_key="labels",
        loss_key="loss"
    )

    runner.train(
        engine=dl.DeviceEngine(),
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        loaders=train_and_valid_loaders,
        callbacks=callbacks,
        verbose=True,
        logdir=logdir,
        num_epochs=config["num_epochs"],
        check=is_debug,
        load_best_on_end=True,
        hparams=config,
        loggers=loggers
    )


if __name__ == '__main__':
    main()
