from stages.taxonomy import Taxonomy
from stages.enrich_and_anotate import EnrichAndAnnotate
from stages.generate import AbstractGenerateStage
from stages.efficacy_filtering_and_potency_measure import DummyEfficacyFilteringAndPotencyMeasure
from stages.refine_dataset import DummyRefineDataset
from stages.classifier import DummyClassifier
import json

# Load the main_config.json
with open('pipeline/main_config.json', 'r') as main_config_file:
    main_config = json.load(main_config_file)

##### Comment out stages which wish or do not wish to run ####

# Taxonomy 
taxonomy_stage = Taxonomy(main_config)
taxonomy_stage.run()

# Generate Stage
AbstractGenerateStage.run(main_config)

# Merge 
AbstractGenerateStage.merge_gen_attacks(main_config)

# # Efficacy Filtering
# efficacy_filtering_and_potency_measure_stage = DummyEfficacyFilteringAndPotencyMeasure(main_config)
# efficacy_filtering_and_potency_measure_stage.run()


##### To be implemented later #####
# # Refine Dataset
# refine_dataset_stage = DummyRefineDataset(
#     main_config,
# )
# refine_dataset_stage.run()

# # Classifier
# classifier_stage = DummyClassifier(
#     main_config,
# )
# classifier_stage.run()