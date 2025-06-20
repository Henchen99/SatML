from abc import ABC, abstractmethod
import os
import json
from dotenv import load_dotenv

load_dotenv()

class AbstractBenchmarkStage(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def execute(self):
        """Execute the classifier stage."""
        pass