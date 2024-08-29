import warnings

class RefineDataset:
    def __init__(self, config):
        self.config = config

    def run(self):
        warnings.warn("RefineDataset stage not implemented", UserWarning)
        print("Skipping RefineDataset stage...")
