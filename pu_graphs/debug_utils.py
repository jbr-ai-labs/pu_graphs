from torch.utils.data import Dataset, Subset


class DebugDataset(Subset):

    def __init__(self,  dataset: Dataset, n_examples: int):
        self.n_examples = min(n_examples, len(dataset))
        super(DebugDataset, self).__init__(dataset=dataset, indices=list(range(self.n_examples)))
