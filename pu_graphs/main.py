import os
from functools import partial
from pathlib import Path

import dgl
import torch
from catalyst.engines import DeviceEngine
from catalyst import dl
from catalyst.metrics import reciprocal_rank, mrr, MRRMetric, AccuracyMetric
from catalyst.utils.config import load_config
from catalyst.utils import set_global_seed
from torch.utils.data import DataLoader

from pu_graphs.data.datasets import DglGraphDataset
from pu_graphs.data.negative_sampling import UniformStrategy
from pu_graphs.data.utils import get_split
from pu_graphs.debug_utils import DebugDataset
from pu_graphs.evaluation.evaluation import compute_score_based_metrics_for_loader, MRRLinkPredictionMetric, \
    AccuracyLinPredictionMetric
from pu_graphs.modeling.dist_mult import DistMult
from pu_graphs.modeling.loss import UnbiasedPULoss, sigmoid_loss, logistic_loss


def main():
    config = load_config("config.yaml")  # FIXME: replace hardcode

    if config["is_debug"]:
        os.environ["WANDB_MODE"] = "dryrun"

    set_global_seed(config["seed"])

    fb15 = dgl.data.FB15k237Dataset()
    full_graph = fb15[0]

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

    if config["is_debug"]:
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
        embedding_dim=config["embedding_dim"]
    )
    optimizer = torch.optim.Adam(model.parameters())
    criterion = UnbiasedPULoss(logistic_loss, pi=0.5)  # FIXME: estimate pi somehow

    def transform_as_pos_neg(batch):
        n_positives = batch["head_indices"].size(0)
        n_unlabeled = batch["neg_head_indices"].size(0)
        new_fields = {
            "head_indices": torch.cat([batch["head_indices"], batch["neg_head_indices"]]),
            "tail_indices": torch.cat([batch["tail_indices"], batch["neg_tail_indices"]]),
            "labels": torch.cat([torch.ones(n_positives), torch.zeros(n_unlabeled)]),
        }
        batch.update(new_fields)
        return batch

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
        )
    ]  # TODO: init

    loggers = {
        "wandb": dl.WandbLogger(project="pu_graphs", name=config["run_name"])
    }

    runner = dl.SupervisedRunner(
        input_key=["head_indices", "tail_indices"],
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
        logdir=Path("./logdir") / config["run_name"],
        num_epochs=config["num_epochs"],
        check=config["is_debug"],
        load_best_on_end=True,
        hparams=config,
        loggers=loggers
    )

    # TODO: move inside runner
    for split_key in ("valid", "test"):
        metrics = compute_score_based_metrics_for_loader(
            graph=graphs[split_key],
            model=model,
            loader=loaders[split_key],
            metrics={
                "mrr": MRRLinkPredictionMetric(topk_args=[full_graph.number_of_nodes()]),
                "acc": AccuracyLinPredictionMetric(topk_args=[1, 3, 5, 10, 20])
            }
        )
        print(split_key, metrics)


if __name__ == '__main__':
    main()
