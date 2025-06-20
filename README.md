# Synthetic Data Generation Pipeline

This repository contains a pipeline for generating synthetic data using various large language models (LLMs). This tool is designed to help you create diverse and realistic datasets for testing, development, or machine learning model training when real-world data is scarce or sensitive.

## Features

* **Multi-Model Support:** Integrates with OpenAI, Llama, Hugging Face, and Together AI for flexible data generation.
* **Configurable Prompts:** Easily customize prompts to guide the synthetic data generation process.
* **Easy to Use:** Simple setup and execution for quick data synthesis.

---

## Setup

To get started with the synthetic data generation pipeline, follow these steps:

### Prerequisites

* Python 3.8+
* pip (Python package installer)

### Environment Variables

Before running the pipeline, you need to set up several environment variables. These variables provide the necessary API keys and configuration for the LLMs.

Create a `.env` file in the root directory of your project, or set these variables directly in your shell environment:

```bash
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
API_VERSION="YOUR_API_VERSION"
DEPLOYMENT_NAME="YOUR_DEPLOYMENT_NAME" 
BASE_URL="YOUR_BASE_URL"
LLAMA_API_KEY="YOUR_LLAMA_API_KEY"
HUGGING_FACE_API_KEY="YOUR_HUGGING_FACE_API_KEY"
TOGETHER_API_KEY="YOUR_TOGETHER_API_KEY"
```

### Installation
Install the required Python libraries using the requirements.txt file:

```bash
pip install -r requirements.txt
```

### Usage
Once you have set up the environment variables and installed the dependencies, you can run the synthetic data generation pipeline.

### Configuration
All pipeline configurations, including model parameters, data generation rules, and output settings, are managed in the YAML file located at:

```bash
pipeline/main_config.yaml
```

Please review and adjust this file according to your data generation needs.

### Running the Pipeline
To execute the pipeline, simply run the following command from the root of the repository:

```bash
python run_pipeline.py
```

This command will initiate the data generation process based on the settings defined in pipeline/main_config.yaml.