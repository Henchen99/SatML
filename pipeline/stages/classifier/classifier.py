import warnings

class Classifier:
    def __init__(self, config):
        self.config = config

    def run(self):
        warnings.warn("Classifier stage not implemented", UserWarning)
        print("Skipping Classifier stage...")
