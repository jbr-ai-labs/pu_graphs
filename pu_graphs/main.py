import sys
from pathlib import Path

import dgl
import hydra_slayer
from pu_graphs.data.datasetPolypharmacy import PolypharmacyDataset
import wandb
from catalyst import dl
from catalyst.utils import set_global_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from pu_graphs.data.datasetWN18RR import WN18RRDataset
from pu_graphs.data.datasets import DglGraphDataset
from pu_graphs.data.negative_sampling import UniformStrategy
from pu_graphs.data.utils import get_split, transform_as_pos_neg
from pu_graphs.debug_utils import DebugDataset
from pu_graphs.evaluation.callback import evaluation_callback
from pu_graphs.external_init_wandb_logger import ExternalInitWandbLogger
from pu_graphs.modeling.complex import ComplEx
from pu_graphs.modeling.dist_mult import DistMult

CONFIG_DIR = Path("config")


def nested_yaml_resolver(name, *, _root_, _parent_, _node_):
    nested_config_path = CONFIG_DIR.joinpath(_node_._key(), name)
    return OmegaConf.load(nested_config_path)


def update_value(oc_config, key: str, value, resolve_anew: bool = True, check_existed: bool = True):
    if check_existed and OmegaConf.select(oc_config, key) is None:
        raise ValueError(f"No value exists for a key = {key}")

    OmegaConf.update(oc_config, key, value)
    if resolve_anew:
        OmegaConf.resolve(oc_config)
    return oc_config


def init_run(config):
    sweep_run = wandb.init(project="pu_graphs", config=config)
    return sweep_run


def main():
    OmegaConf.register_new_resolver("nested_yaml", nested_yaml_resolver)
    oc_config = OmegaConf.load(CONFIG_DIR / "config.yaml")
    OmegaConf.resolve(oc_config)

    # Merge with cli parameters, you can set nested values via dot notation, e.g criterion.pi=0.99
    oc_config.merge_with_dotlist([
        x[2:]  # Drop '--' prefix
        for x in sys.argv[1:]
    ])

    plain_config = OmegaConf.to_container(oc_config, resolve=True)

    config = hydra_slayer.get_from_params(**plain_config)
    is_debug = config["is_debug"]

    #if is_debug:
    #    os.environ["WANDB_MODE"] = "dryrun"

    set_global_seed(config["seed"])

    if config['dataset'] == 'FB15k237':
        dataset = dgl.data.FB15k237Dataset()
    elif config['dataset'] == 'WN18RR':
        dataset = WN18RRDataset()
    elif config['dataset'] == 'Polypharmacy':
        dataset = PolypharmacyDataset()
    else:
        print(f"No such dataset as {config['dataset']}")
        return
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
    model = None
    if config['model'] == 'distmult':
        model = DistMult(
            n_nodes=full_graph.number_of_nodes(),
            n_relations=graphs["train"].edata["etype"].max().item() + 1,
            embedding_dim=config["embedding_dim"]
        )
    elif config['model'] == 'complex':
        model = ComplEx(
            n_nodes=full_graph.number_of_nodes(),
            n_relations=graphs["train"].edata["etype"].max().item() + 1,
            embedding_dim=config["embedding_dim"]
        )
    else:
        print(f"No such model as {config['model']}")
        return

    optimizer = config["optimizer"](model.parameters())
    criterion = config["criterion"]

    wandb_run = init_run(config=plain_config)
    run_name = wandb_run.name or config["run_name"]
    logdir = Path("./logdir") / run_name
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
            logdir=logdir.joinpath("checkpoints").__str__(),
            loader_key="valid",
            metric_key="loss",
            minimize=True,
            load_on_stage_end="best"
        ),
        evaluation_callback(
            graphs=graphs,
            loaders=loaders,
            eval_loader_key="valid",
            is_debug=is_debug
        ),
        evaluation_callback(
            graphs=graphs,
            loaders=loaders,
            eval_loader_key="test",
            is_debug=is_debug
        )
    ]

    loggers = {
        "wandb": ExternalInitWandbLogger(wandb_run)
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
        loggers=loggers
    )


if __name__ == '__main__':
    main()
