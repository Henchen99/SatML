from stages.taxonomy import DummyTaxonomy
from stages.enrich_and_anotate import EnrichAndAnnotate
from stages.generate import GenerateHC
from stages.efficacy_filtering_and_potency_measure import DummyEfficacyFilteringAndPotencyMeasure
from stages.refine_dataset import DummyRefineDataset
from stages.classifier import DummyClassifier
import json

with open('config.json') as config_file:
    config = json.load(config_file)

# Taxonomy 
taxonomy_stage = DummyTaxonomy(
    config,
)
taxonomy_stage.run()


# Enrich and Annotate
enrich_and_annotate_stage = EnrichAndAnnotate(
    config,
    input_csv_file_path='data/collected_data/collected_data.csv',
    output_csv_file_path='data/annotated_output/collected_annotated.csv'
)
enrich_and_annotate_stage.run()

# Generate
generate_stage = GenerateHC(
    config,
    generated_attack_csv_file_path="data/generated_attacks/generated_attacks.csv"
)
generate_stage.run()

# Efficacy Filtering
efficacy_filtering_and_potency_measure_stage = DummyEfficacyFilteringAndPotencyMeasure(
    config,
)
efficacy_filtering_and_potency_measure_stage.run()

# Refine Dataset
refine_dataset_stage = DummyRefineDataset(
    config,
)
refine_dataset_stage.run()

# Classifier
classifier_stage = DummyClassifier(
    config,
)
classifier_stage.run()