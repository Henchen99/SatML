# SatML

## Configuration Setup

Before running the code, **each implemented generation method needs to set up their own configuration file**, `generationMethod_config.json`. This file contains settings specific to the version's environment and is required for the code to run correctly. 

### Format

The `generationMethod_config.json` file should be in the following format:

```json
{
    "sampled_data_json_file_path": "path/to/annotated/data.json",
    "generated_attack_json_file_path": "path/to/generated/attacks.json",
    "generation_strat": "VERSION"
}
```

**In addition to setting up each config file, a .env file is required to be set up in the root directory of the repository** This should contain the API key(s), BASE_URL etc. in order to run the pipeline.

### Format

The `.env` file should be in the following format and filled in where necessary:

```json
API_KEY="YOUR_KEY_HERE"
TOGETHER_API_KEY="YOUR_KEY_HERE"
LLAMA_API_KEY="YOUR_KEY_HERE"
HUGGING_FACE_API_KEY="YOUR_KEY_HERE"
BASE_URL="YOUR_URL_HERE"
API_VERSION="YOUR_API_VERSION_HERE"
DEPLOYMENT_NAME="YOUR_DEPLOYMENT_NAME_HERE"
```
