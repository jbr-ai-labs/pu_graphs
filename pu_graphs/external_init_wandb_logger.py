from catalyst import dl


class ExternalInitWandbLogger(dl.WandbLogger):

    # noinspection PyMissingConstructor
    def __init__(self, run):
        self.run = run
