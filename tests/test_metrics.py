import typing as ty
from collections import OrderedDict
from copy import copy

import numpy as np
import pytest
import sparse
import torch

from pu_graphs.evaluation import MRRLinkPredictionMetric, FilteredLinkPredictionMetric
# TODO: think about generating input data
from pu_graphs.evaluation.evaluation import AdjustedMeanRankIndex

TEST_METRIC_DATA = OrderedDict([
    ("n_nodes", 5),
    ("head_idx", torch.tensor([1, 2, 3, 4])),
    ("tail_idx", torch.tensor([0, 0, 1, 2])),
    ("relation_idx", torch.tensor([0, 0, 1, 1])),
    ("logits", torch.tensor(
        [
            [10, -1, 5, -10, 2],  # rank 1
            [5, 10, 3, 2, -1],  # rank 2
            [10, -10, 9, -20, 30],  # rank 4
            [5, 4, 3, 1, 1]  # rank 3
        ]
    )),
])


FULL_ADJ_MAT = sparse.as_coo(np.array([
    [
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],  # shift by 0
        [1, 1, 0, 0, 0],  # shift rank by 1 up
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1],  # shift rank by 2 up
        [1, 0, 1, 0, 0],  # shift rank by 1 up
    ]
]))


def parametrize_with_dict(dict: ty.OrderedDict):
    return pytest.mark.parametrize(
        argnames=list(dict.keys()),
        argvalues=[list(dict.values())]
    )


@parametrize_with_dict(TEST_METRIC_DATA)
class TestMetrics:

    @staticmethod
    def _get_rank(scores: ty.List[int], position: int):
        argsorted_scores = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return argsorted_scores.index(position) + 1

    @staticmethod
    def _get_rank_filtered(scores: ty.List[int], position: int, adj_row: ty.List[int]):
        adj_row = copy(adj_row)
        assert adj_row[position] == 1
        adj_row[position] = 0

        scores = [
            s if adj_row[i] == 0 else float("-inf")
            for i, s in enumerate(scores)
        ]

        return TestMetrics._get_rank(scores, position)

    @staticmethod
    def _get_actual_ranks(logits, tail_idx):
        return [
            TestMetrics._get_rank(row.tolist(), pos.item())
            for row, pos in zip(logits, tail_idx)
        ]

    @staticmethod
    def _get_actual_ranks_filtered(logits, head_idx, tail_idx, relation_idx, full_adj_mat):
        adj_mat = full_adj_mat[relation_idx, head_idx].todense()
        return [
            TestMetrics._get_rank_filtered(row.tolist(), pos.item(), adj_row)
            for row, pos, adj_row in zip(logits, tail_idx, adj_mat)
        ]

    @staticmethod
    def _get_actual_mrr(actual_ranks: ty.List[int]):
        return sum(map(lambda r: 1 / r, actual_ranks)) / len(actual_ranks)

    def test_mrr(self, n_nodes, head_idx, tail_idx, relation_idx, logits):
        mrr = MRRLinkPredictionMetric(topk_args=[1, n_nodes])

        mrr.update(
            head_idx=head_idx, tail_idx=tail_idx, relation_idx=relation_idx, logits=logits
        )

        result = mrr.compute_key_value()

        actual_ranks = TestMetrics._get_actual_ranks(logits, tail_idx)
        actual_mrr = TestMetrics._get_actual_mrr(actual_ranks)

        assert result[f"mrr{n_nodes:02d}"] == pytest.approx(actual_mrr)

    @pytest.mark.parametrize("full_adj_mat", [FULL_ADJ_MAT])
    def test_mrr_filtered(self, head_idx, tail_idx, relation_idx, logits, n_nodes, full_adj_mat):
        mrr = FilteredLinkPredictionMetric(
            MRRLinkPredictionMetric(topk_args=[1, n_nodes]),
            full_adj_mat=full_adj_mat
        )

        mrr.update(
            head_idx=head_idx, tail_idx=tail_idx, relation_idx=relation_idx, logits=logits
        )

        result = mrr.compute_key_value()

        actual_ranks = TestMetrics._get_actual_ranks_filtered(
            logits=logits, head_idx=head_idx, tail_idx=tail_idx, relation_idx=relation_idx, full_adj_mat=full_adj_mat
        )
        actual_mrr = TestMetrics._get_actual_mrr(actual_ranks)

        assert result[f"mrr{n_nodes:02d}"] == pytest.approx(actual_mrr)

    @pytest.mark.parametrize("full_adj_mat", [FULL_ADJ_MAT])
    def test_filtered_metric_is_greater_or_equal(self, head_idx, tail_idx, relation_idx, logits, n_nodes, full_adj_mat):
        topk_args = [1, n_nodes]
        mrr = MRRLinkPredictionMetric(topk_args=topk_args)
        filtered_mrr = FilteredLinkPredictionMetric(
            MRRLinkPredictionMetric(topk_args=topk_args),
            full_adj_mat=full_adj_mat
        )

        mrr.update(
            head_idx=head_idx, tail_idx=tail_idx, relation_idx=relation_idx, logits=logits
        )
        filtered_mrr(
            head_idx=head_idx, tail_idx=tail_idx, relation_idx=relation_idx, logits=logits
        )

        result = mrr.compute_key_value()
        filtered_result = filtered_mrr.compute_key_value()

        for key, filtered_value in filtered_result.items():
            value = result[key]
            if key.endswith("/std") or filtered_result[key] == pytest.approx(value):
                continue
            assert filtered_value > value

    # TODO: add tests for metrics other than MRR

    def test_amri(self, n_nodes, head_idx, tail_idx, relation_idx, logits):
        amri = AdjustedMeanRankIndex(topk_args=[n_nodes])

        amri.update(
            head_idx=head_idx, tail_idx=tail_idx, relation_idx=relation_idx, logits=logits
        )

        r = amri.compute_key_value()
        assert r["amri"] == pytest.approx(0.25)

    def test_amri_batched(self, n_nodes, head_idx, tail_idx, relation_idx, logits):
        amri = AdjustedMeanRankIndex(topk_args=[n_nodes])

        split = 2
        amri.update(
            head_idx=head_idx[:split], tail_idx=tail_idx[:split], relation_idx=relation_idx, logits=logits[:split]
        )
        amri.update(
            head_idx=head_idx[split:], tail_idx=tail_idx[split:], relation_idx=relation_idx, logits=logits[split:]
        )

        r = amri.compute_key_value()
        assert r["amri"] == pytest.approx(0.25)
