from abc import ABC, abstractmethod
import os
import json
from dotenv import load_dotenv

load_dotenv()

class AbstractDataRefinementStage(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def execute(self):
        """Execute the data_refinement stage."""
        pass