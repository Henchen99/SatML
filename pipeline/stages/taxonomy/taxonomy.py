import warnings

class Taxonomy:
    def __init__(self, config):
        self.config = config

    def run(self):
        warnings.warn("Taxonomy stage not implemented", UserWarning)
        print("Skipping Taxonomy stage...")
