# pipeline_manager.py

import sys
import os
import json
import logging
import importlib
import inspect
from pathlib import Path
import traceback  # Import traceback module for detailed error information

# Add the parent directory to sys.path to ensure 'pipeline' is discoverable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.stages._1_taxonomy.taxonomy import TaxonomyStage
from pipeline.stages._2_generate.base_generate import AbstractGenerateStage
from pipeline.stages._3_evaluate.evaluate import EvaluateStage
from pipeline.stages._4_data_refinement.data_refining import DataRefinementStage
from pipeline.stages._5_classifier.classifier import ClassifierStage
from pipeline.stages._6_benchmark.benchmark import BenchmarkStage

# Configure logging to include DEBUG level and format tracebacks
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG for more verbose output
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config_path='pipeline/main_config.json'):
        self.config_path = config_path
        self.stages = []
        # self.print_sys_path()  # Print sys.path for debugging
        # self.print_cwd()        # Print current working directory for debugging
        self.load_config()

    # def print_sys_path(self):
    #     logger.debug("Current sys.path:")
    #     for path in sys.path:
    #         logger.debug(f"  {path}")

    # def print_cwd(self):
    #     cwd = os.getcwd()
    #     logger.debug(f"Current Working Directory: {cwd}")

    def load_config(self):
        logger.info(f"Loading pipeline configuration from {self.config_path}")
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.config = config
            # logger.debug(f"Pipeline configuration loaded: {config}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.debug(traceback.format_exc())  # Log full traceback
            raise

        # Taxonomy Stage
        if config.get('taxonomy', {}).get('enabled', False):
            taxonomy_config = config['taxonomy']['config_path']
            try:
                taxonomy_stage = TaxonomyStage(config={'config_path': taxonomy_config})
                self.stages.append(taxonomy_stage)
                logger.info("#### Taxonomy Stage enabled and added to pipeline #### ")
            except Exception as e:
                logger.error(f"Failed to initialize Taxonomy Stage: {e}")
                logger.debug(traceback.format_exc())  # Log full traceback

        # Generate Stage - Dynamic Loading
        if config.get('generate', {}).get('enabled', False):
            generate_config = config['generate']
            # Exclude 'enabled' and 'default_config' from being treated as stages
            generate_stages = {k: v for k, v in generate_config.items() if k not in ['enabled', 'default_config']}
            generate_dir = Path('pipeline/stages/_2_generate')

            try:
                generate_contents = os.listdir(generate_dir)
                # logger.info(f"Contents of generate directory: {generate_contents}")
            except Exception as e:
                logger.error(f"Failed to list generate directory '{generate_dir}': {e}")
                logger.debug(traceback.format_exc())
                raise

            for stage_name, stage_info in generate_stages.items():
                if not stage_info.get('enabled', False):
                    logger.info(f"Generate stage '{stage_name}' is disabled. Skipping.")
                    continue
                stage_config_path = stage_info.get('config_path')
                if not stage_config_path:
                    logger.warning(f"Missing 'config_path' for generate stage '{stage_name}'. Skipping.")
                    continue
                try:
                    stage_dir = generate_dir / stage_name
                    try:
                        stage_dir_contents = os.listdir(stage_dir)
                        # logger.info(f"Contents of '{stage_name}' directory: {stage_dir_contents}")
                    except Exception as e:
                        logger.error(f"Failed to list directory '{stage_dir}': {e}")
                        logger.debug(traceback.format_exc())
                        continue  # Skip this stage if directory listing fails

                    with open(stage_config_path, 'r') as f:
                        stage_stage_config = json.load(f)

                    # Merge main generate's default_config with stage's stage_config
                    combined_config = {
                        'default_config': generate_config.get('default_config', {}),
                        'stage_config': stage_stage_config
                    }

                    module_path = f"pipeline.stages._2_generate.{stage_name}.generate"
                    logger.info(f"Attempting to import module: {module_path}")

                    # Dynamically import the generator stage module
                    stage_module = importlib.import_module(module_path)

                    # Find the class that inherits from AbstractGenerateStage
                    stage_class = None
                    for name, obj in inspect.getmembers(stage_module, inspect.isclass):
                        if issubclass(obj, AbstractGenerateStage) and obj is not AbstractGenerateStage:
                            stage_class = obj
                            break
                    if stage_class is None:
                        raise ImportError(f"No subclass of AbstractGenerateStage found in module '{module_path}'.")

                    # Initialize stage with combined_config
                    generator_stage = stage_class(config=combined_config)
                    self.stages.append(generator_stage)
                    logger.info(f"#### Generator Stage '{stage_name}' enabled and added to pipeline ####")
                except ModuleNotFoundError as e:
                    logger.error(f"Module '{module_path}' not found for generate stage '{stage_name}'.")
                    logger.debug(traceback.format_exc())  # Log full traceback
                except ImportError as ie:
                    logger.error(f"Import error for generate stage '{stage_name}': {ie}")
                    logger.debug(traceback.format_exc())  # Log full traceback
                except Exception as e:
                    logger.error(f"Failed to initialize generate stage '{stage_name}': {e}")
                    logger.debug(traceback.format_exc())  # Log full traceback

        # Evaluate Stage
        if config.get('evaluator', {}).get('enabled', False):
            evaluator_config = config['evaluator']
            try:
                evaluate_stage = EvaluateStage(config={'config_path': evaluator_config})
                self.stages.append(evaluate_stage)
                logger.info("#### Evaluate Stage enabled and added to pipeline ####")
            except Exception as e:
                logger.error(f"Failed to initialize Evaluate Stage: {e}")
                logger.debug(traceback.format_exc())  # Log full traceback

        # Data Refinement Stage
        if config.get('data_refinement', {}).get('enabled', False):
            data_refinement_config = config['data_refinement']
            try:
                data_refinement_stage = DataRefinementStage(config={'config_path': data_refinement_config})
                self.stages.append(data_refinement_stage)
                logger.info("#### Data Refinement Stage enabled and added to pipeline ####")
            except Exception as e:
                logger.error(f"Failed to initialize Data Refinement Stage: {e}")
                logger.debug(traceback.format_exc())


        # Classifier Stage
        if config.get('classifier', {}).get('enabled', False):
            classifier_config = config['classifier']
            try:
                classifier_stage = ClassifierStage(config={'config_path': classifier_config})
                self.stages.append(classifier_stage)
                logger.info("#### Classifier Stage enabled and added to pipeline #####")
            except Exception as e:
                logger.error(f"Failed to initialize Classifier Stage: {e}")
                logger.debug(traceback.format_exc())  # Log full traceback

        # Benchmark Stage
        if config.get('benchmark', {}).get('enabled', False):
            benchmark_config = config['benchmark']
            try:
                benchmark_stage = BenchmarkStage(config=benchmark_config)
                self.stages.append(benchmark_stage)
                logger.info("#### Benchmark Stage enabled and added to pipeline #####")
            except Exception as e:
                logger.error(f"Failed to initialize Benchmark Stage: {e}")
                logger.debug(traceback.format_exc())  # Log full traceback

    def run(self):
        logger.info("===============================================")
        logger.info("Starting pipeline execution")
        for stage in self.stages:
            try:
                logger.info("===============================================")
                logger.debug(f"Running stage: {stage.__class__.__name__}")
                stage.execute()
                logger.info(f"Stage {stage.__class__.__name__} completed successfully. ✅")
            except Exception as e:
                logger.error(f"Error running stage {stage.__class__.__name__}: {e} ❌")
                logger.debug(traceback.format_exc())  # Log full traceback
        logger.info("😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊")
        logger.info("Pipeline execution completed. 💯")
        logger.info("😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊")

    def merge_generated_attacks(self):
        """Merge all generated attacks into a combined JSON file."""
        try:
            logger.info("Starting merge of generated attacks.")
            AbstractGenerateStage.merge_gen_attacks(self.config)
            logger.info("Successfully merged all generated attacks.")
        except Exception as e:
            logger.error(f"Failed to merge generated attacks: {e}")
            logger.debug(traceback.format_exc())  # Log full traceback

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
    logger.info("Running merge_generated_attacks method.")
    pipeline.merge_generated_attacks()
    logger.info("Finished running merge_generated_attacks method.")