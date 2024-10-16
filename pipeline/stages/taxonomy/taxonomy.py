import warnings
from abc import ABC, abstractmethod
import hashlib
import json

class AbstractTaxonomyStage(ABC):
    @abstractmethod
    def run(self):
        """
        Run the Taxonomy stage.
        """
        pass


    def hashing(self, data):
        '''
        Assigns each row in the raw dataset with a unique SHA-256 number         
        '''
        hashed_data = []
        for row in data:
            # check if id is already hashed
            if len(str(row["SHA-256"])) == 64:
                hashed_data.append(row)
            else:
                # Create a new SHA-256 hash object
                hash_object = hashlib.sha256()

                # Update the hash object with the row's ID and prompt
                hash_object.update(str(row["id"]).encode("utf-8"))
                hash_object.update(row["prompt"].encode("utf-8"))

                # Get the hexadecimal representation of the hash
                row["id"] = hash_object.hexdigest()

                hashed_data.append(row)

        return hashed_data



class DummyTaxonomy(AbstractTaxonomyStage):
    def __init__(self, config):
        self.config = config

    def run(self):

        with open('pipeline/data/collected_data/collected_data.json', 'r') as data_file:
            data = json.load(data_file)
            
            # Hash the data
            hashed_data = self.hashing(data)
            
            # Save the hashed data to a new file (optional)
            with open('pipeline/data/collected_data/collected_data.json', 'w') as output_file:
                json.dump(hashed_data, output_file, indent=4)

        warnings.warn("Taxonomy stage not implemented", UserWarning)
        print("Skipping Taxonomy stage...")