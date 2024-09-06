from stages.taxonomy import DummyTaxonomy
from stages.enrich_and_anotate import EnrichAndAnnotate
import inspect
import importlib
from stages.generate import AbstractGenerateStage
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
    input_json_file_path='data/collected_data/collected_data.json',
    output_json_file_path='data/annotated_output/collected_annotated.json'
)
enrich_and_annotate_stage.run()

# Generate Stage
def get_generate_stages():
    generate_module = importlib.import_module("stages.generate")

    # Find all subclasses of AbstractGenerateStage in the generate module
    generate_stages = []
    for name, obj in inspect.getmembers(generate_module):
        if inspect.isclass(obj) and issubclass(obj, AbstractGenerateStage) and obj != AbstractGenerateStage:
            generate_stages.append(obj)
    return generate_stages

# Load and instantiate all generate stages 
def run_all_generate_stages(config):
    generate_classes = get_generate_stages()

    for GenerateClass in generate_classes:
        stage_instance = GenerateClass(config, f"data/generated_attacks/generated_attacks.json")
        stage_instance.run()

# Run the function to execute all stages
run_all_generate_stages(config)



##### To be implemented later #####

# # Efficacy Filtering
# efficacy_filtering_and_potency_measure_stage = DummyEfficacyFilteringAndPotencyMeasure(
#     config,
# )
# efficacy_filtering_and_potency_measure_stage.run()

# # Refine Dataset
# refine_dataset_stage = DummyRefineDataset(
#     config,
# )
# refine_dataset_stage.run()

# # Classifier
# classifier_stage = DummyClassifier(
#     config,
# )
# classifier_stage.run()