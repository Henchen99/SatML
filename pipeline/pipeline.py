from stages.taxonomy import DummyTaxonomy
from stages.enrich_and_anotate import EnrichAndAnnotate
import inspect
import importlib
from stages.generate import AbstractGenerateStage
from stages.efficacy_filtering_and_potency_measure import DummyEfficacyFilteringAndPotencyMeasure
from stages.refine_dataset import DummyRefineDataset
from stages.classifier import DummyClassifier
import json

# Load the main_config.json
with open('pipeline/main_config.json', 'r') as main_config_file:
    main_config = json.load(main_config_file)

# Taxonomy 
taxonomy_stage = DummyTaxonomy(
    main_config,
)
taxonomy_stage.run()

# Generate Stage
AbstractGenerateStage.run(main_config)

# Merge 
AbstractGenerateStage.merge_gen_attacks(main_config)

##### To be implemented later #####

# # Efficacy Filtering
# efficacy_filtering_and_potency_measure_stage = DummyEfficacyFilteringAndPotencyMeasure(
#     main_config,
# )
# efficacy_filtering_and_potency_measure_stage.run()

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