from abc import ABC, abstractmethod
import os
import json
from dotenv import load_dotenv

load_dotenv()

class AbstractEvaluateStage(ABC):
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv('API_KEY')

    @abstractmethod
    def execute(self):
        """Execute the generation pipeline."""
        pass