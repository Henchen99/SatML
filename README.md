# SatML

Before running the code, each implemented version needs to set up their own configuration file, version_config.json. This file contains settings specific to the version's environment and is required for the code to run correctly.

Format

The user_config.json file should be in the following format:
JSON
{
    "sampled_data_json_file_path": "path/to/annotated/data.json",
    "generated_attack_json_file_path": "path/to/generated/attacks.json",
    "generation_strat": "VERSION"
}
