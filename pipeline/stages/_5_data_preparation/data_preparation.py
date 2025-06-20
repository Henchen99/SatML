from .base_data_preparation import AbstractDataPreparationStage
from .data_split import DataSplitStage
from .data_clustering import DataClusteringStage
from .data_labelling import DataLabellingStage  
from .data_assembler import DataAssemblerStage

import json
import yaml
from dotenv import load_dotenv
import os

load_dotenv()

class DataPreparationStage(AbstractDataPreparationStage):
    def __init__(self, config):
        super().__init__(config)

    def execute(self):
        print("[*] Starting Data Preparation Stage")

        print("[*] Running Labeling Stage")
        DataLabellingStage(self.config).execute()

        print("[*] Running Clustering Stage")
        DataClusteringStage(self.config).execute()

        print("[*] Running Assembling Stage")
        DataAssemblerStage(self.config).execute()

        print("[*] Running Data Split Stage")
        DataSplitStage(self.config).execute()

        print("[✔] Data Preparation Stage Completed")