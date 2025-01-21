# run_pipeline.py
from pipeline_manager import Pipeline

def main():
    pipeline = Pipeline(config_path='pipeline/main_config.json')
    pipeline.run()
    pipeline.merge_generated_attacks()

if __name__ == "__main__":
    main()
