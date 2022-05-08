import os
import sys
from pathlib import Path
from typing import Dict

import dgl
import hydra_slayer
import wandb
from catalyst import dl
from catalyst.utils import set_global_seed
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from pu_graphs.data import keys
from pu_graphs.data.datasetWN18RR import WN18RRDataset
from pu_graphs.data.datasets import DglGraphDataset
from pu_graphs.data.unlabeled_sampler import UnlabeledSampler
from pu_graphs.data.utils import get_split
from pu_graphs.debug_utils import DebugDataset
from pu_graphs.evaluation.callback import evaluation_callback
from pu_graphs.external_init_wandb_logger import ExternalInitWandbLogger
from pu_graphs.modeling.complex import ComplEx
from pu_graphs.modeling.dist_mult import DistMult
from pu_graphs.modeling.pan_runner import PanRunner, get_pan_loss_by_key, LearnableLogitToProbability, \
    SigmoidLogitToProbability

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


def init_opt_callbacks(key: str) -> Dict[str, dl.Callback]:
    if key == PanRunner.DISC_KEY:
        metric_key = PanRunner.LOSS_DISC_KEY
        input_key = [keys.disc_positive_probs, keys.disc_unlabeled_probs, keys.cls_unlabeled_probs]
    elif key == PanRunner.CLS_KEY:
        metric_key = PanRunner.LOSS_CLS_KEY
        input_key = [keys.disc_unlabeled_probs, keys.cls_unlabeled_probs]
    else:
        raise ValueError(f"key: {key}")

    # noinspection PyTypeChecker
    callbacks = {
        PanRunner.criterion_key(key): ManualCriterionCallback(
            input_key=input_key,
            target_key=[],
            metric_key=metric_key,
            criterion_key=key
        ),
        PanRunner.optimizer_key(key): ManualOptimizerCallback(
            metric_key=metric_key,
            model_key=key,
            optimizer_key=key
        )
    }

    return callbacks


def init_checkpoint_and_early_stop_callbacks(config, logdir: Path) -> Dict[str, dl.Callback]:
    eval_metric_key: str = config["eval_metric_key"]
    if eval_metric_key.startswith("loss"):
        eval_metric_minimize = True
    elif eval_metric_key.startswith("mrr"):
        eval_metric_minimize = False
    else:
        raise ValueError(f"Not sure if eval metric should be maximized or minimized: eval_metric_key={eval_metric_key}")
    print(f"Debug: eval_metric_key={eval_metric_key}, eval_metric_minimize={eval_metric_minimize}")

    return {
        "early_stopping": dl.EarlyStoppingCallback(
            patience=config["patience"],
            loader_key="valid",
            metric_key=eval_metric_key,
            minimize=eval_metric_minimize
        ),
        "checkpoint": dl.CheckpointCallback(
            logdir=logdir.joinpath("checkpoints").__str__(),
            loader_key="valid",
            metric_key=eval_metric_key,
            minimize=eval_metric_minimize,
            load_on_stage_end="best"
        ),
    }


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

    if is_debug:
        os.environ["WANDB_MODE"] = "dryrun"

    set_global_seed(config["seed"])

    if config['dataset'] == 'FB15k237':
        dataset = dgl.data.FB15k237Dataset()
    elif config['dataset'] == 'WN18RR':
        dataset = WN18RRDataset()
    else:
        print(f"No such dataset as {config['dataset']}")
        return
    full_graph = dataset[0]

    graphs = {
        k: get_split(full_graph, k)
        for k in ("train", "valid", "test")
    }
    datasets = {
        split_key: DglGraphDataset(graph=graph)
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

    def model_builder(*args, **kwargs):
        if config['model'] == 'distmult':
            model_cls = DistMult
        elif config['model'] == 'complex':
            model_cls = ComplEx
        else:
            raise ValueError(f"No such model as {config['model']}")

        # debug print:
        if max_norm := kwargs.get("max_norm") is not None:
            print(f"max_norm = f{max_norm}")

        model = model_cls(*args, **kwargs)

        logit_to_prob = "logit_to_prob"
        if config[logit_to_prob] == "learnable":
            return LearnableLogitToProbability(model)
        elif config[logit_to_prob] == "sigmoid":
            return SigmoidLogitToProbability(model)
        raise ValueError(f"Unsupported logit_to_prob value: {logit_to_prob}")

    pan_keys = [PanRunner.DISC_KEY, PanRunner.CLS_KEY]
    # Same models for disc and cls
    model = nn.ModuleDict({
        k: model_builder(
            n_nodes=full_graph.number_of_nodes(),
            n_relations=graphs["train"].edata["etype"].max().item() + 1,
            embedding_dim=config["embedding_dim"],
            max_norm=config.get("max_norm", None)
        ) for k in pan_keys
    })

    optimizer = {
        k: config["optimizer"](model[k].parameters())
        for k in pan_keys
    }

    criterion = {
        k: get_pan_loss_by_key(k, config["alpha"])
        for k in pan_keys
    }

    wandb_run = init_run(config=plain_config)
    run_name = wandb_run.name or config["run_name"]
    logdir = Path("./logdir") / run_name

    callbacks = {
        **init_opt_callbacks(PanRunner.DISC_KEY),
        **init_opt_callbacks(PanRunner.CLS_KEY),
        **init_checkpoint_and_early_stop_callbacks(config, logdir),
        "valid_eval": evaluation_callback(
            graphs=graphs,
            loaders=loaders,
            eval_loader_key="valid",
            is_debug=is_debug,
            model_key=PanRunner.DISC_KEY,
            eval_every_epoch=config["eval_every_epoch"]
        ),
        "test_eval": evaluation_callback(
            graphs=graphs,
            loaders=loaders,
            eval_loader_key="test",
            is_debug=is_debug,
            model_key=PanRunner.DISC_KEY
        )
    }

    loggers = {
        "wandb": ExternalInitWandbLogger(wandb_run)
    }

    runner = PanRunner(unlabeled_sampler=UnlabeledSampler(graph=graphs["train"]))

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


class ManualCriterionCallback(dl.CriterionCallback):

    def on_batch_end(self, runner: "IRunner") -> None:
        return

    def on_batch_end_manual(self, runner) -> None:
        super(ManualCriterionCallback, self).on_batch_end(runner)


class ManualOptimizerCallback(dl.OptimizerCallback):

    def on_batch_end(self, runner: "IRunner"):
        return

    def on_batch_end_manual(self, runner) -> None:
        super(ManualOptimizerCallback, self).on_batch_end(runner)


if __name__ == '__main__':
    main()
