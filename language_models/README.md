# Language Models

This directory contains a flexible language model system that supports multiple LLM providers. The system prioritizes environment variables for configuration, making it secure and easy to manage across different environments.

## Supported Providers

- **OpenAI**: Standard OpenAI API
- **Azure OpenAI**: Microsoft Azure OpenAI Service
- **Llama3**: Local or remote Llama3 endpoints

## Architecture

### Base Class
All language models inherit from `LanguageModel` which provides:
- Consistent interface with `generate()` method
- Environment-first configuration loading
- Configuration validation
- Logging

### Factory Pattern
The `LanguageModelFactory` creates model instances based on configuration and supports:
- Dynamic provider registration
- Error handling
- Provider validation

## Configuration

### Environment Variables (.env file)

**This is the primary and recommended way to configure the system.** Create a `.env` file in your project root with the appropriate variables for your provider:

#### For OpenAI
```bash
OPENAI_API_KEY=your-openai-api-key
```

#### For Azure OpenAI
```bash
OPENAI_API_KEY=your-azure-openai-api-key
BASE_URL=https://your-resource.openai.azure.com/
DEPLOYMENT_NAME=gpt-4o-mini
API_VERSION=2023-07-01-preview
```

#### For Llama3
```bash
LLAMA3_BASE_URL=http://localhost:10001/v1
LLAMA3_API_KEY=your-llama3-api-key  # Optional
```

### Configuration Priority

The system follows this priority order:
1. **Environment Variables** (recommended)
2. **Config Dictionary** (fallback)
3. **Default Values** (where applicable)

### Provider Configuration

#### OpenAI Configuration
```yaml
engine: openai
model: gpt-4o-mini
max_tokens: 4096
temperature: 1.0
# All sensitive config loaded from environment variables
```

#### Azure OpenAI Configuration
```yaml
engine: azure
model: gpt-4o-mini  
max_tokens: 4096
temperature: 1.0
# All sensitive config loaded from environment variables
```

#### Llama3 Configuration
```yaml
engine: llama3
model: meta/llama-3.1-8b-instruct
max_tokens: 4096
temperature: 0.7
# All sensitive config loaded from environment variables
```

## Usage

### Basic Usage
```python
from language_models.language_model_selection import LanguageModelFactory

# Only non-sensitive configuration needed
config = {
    "engine": "azure",  # or openai, llama3
    "model": "gpt-4o-mini",
    "max_tokens": 1000,
    "temperature": 0.7
}

# All sensitive config (API keys, endpoints) loaded from environment
model = LanguageModelFactory.create_model(config)

# Generate text
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
]

response = model.generate(messages)
print(response)
```

### Pipeline Integration
Configure your provider in `pipeline/main_config.yaml`:

```yaml
generate:
  default_config:
    engine: azure  # or openai, llama3
    model: gpt-4o-mini
    # All sensitive config loaded from .env file automatically
```

## Switching Between Providers

Simply change the `engine` parameter in your configuration and ensure the appropriate environment variables are set:

1. **To use OpenAI**: Set `engine: openai` and ensure `OPENAI_API_KEY` is in your `.env`
2. **To use Azure**: Set `engine: azure` and ensure `OPENAI_API_KEY`, `BASE_URL`, `DEPLOYMENT_NAME`, `API_VERSION` are in your `.env`
3. **To use Llama3**: Set `engine: llama3` and ensure `LLAMA3_BASE_URL` is in your `.env`

## Adding New Providers

### Step 1: Create Provider Class
Create a new file in the `language_models/` directory:

```python
# language_models/my_provider.py
from typing import List, Dict, Any
from .base_language_models import LanguageModel

class MyProvider(LanguageModel):
    def _set_config(self) -> None:
        """Set provider-specific configuration."""
        # Define environment variable mapping
        env_mapping = {
            'api_key': 'MY_PROVIDER_API_KEY',
            'endpoint': 'MY_PROVIDER_ENDPOINT'
        }
        
        # Validate required fields
        self._validate_config(['api_key', 'endpoint'], env_mapping)
        
        # Load from environment variables first
        self.api_key = self._get_env_var('MY_PROVIDER_API_KEY', 'api_key')
        self.endpoint = self._get_env_var('MY_PROVIDER_ENDPOINT', 'endpoint')

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Generate text using your provider's API."""
        # Implement your provider's API call
        pass
```

### Step 2: Register Provider
Add your provider to the factory:

```python
# language_models/language_model_selection.py
from .my_provider import MyProvider

class LanguageModelFactory:
    _providers = {
        # ... existing providers
        'myprovider': MyProvider,
    }
```

### Step 3: Configure Environment Variables
Add the required environment variables to your `.env` file.

## Error Handling

The system provides comprehensive error handling:
- **Configuration Validation**: Missing required environment variables are caught early
- **Network Errors**: Proper timeout and retry handling
- **API Errors**: Meaningful error messages with logging
- **Graceful Degradation**: Failed requests don't crash the pipeline

## Best Practices

1. **Use Environment Variables**: Store all sensitive data in `.env` files
2. **Version Control**: Add `.env` to `.gitignore` to avoid committing secrets
3. **Set Appropriate Timeouts**: Configure timeouts based on your provider's expected response time
4. **Monitor Logs**: Enable logging to track API calls and errors
5. **Test Configurations**: Validate your environment variables before running the pipeline

## Troubleshooting

### Common Issues

**Missing Environment Variables**
```
ValueError: Missing required configuration. Set environment variables or add to config: ['api_key (env: OPENAI_API_KEY)']
```
Solution: Add the missing environment variable to your `.env` file.

**Connection Timeout**
```
requests.exceptions.RequestException: Network error calling Azure OpenAI
```
Solution: Check your network connection and increase the timeout value in config.

**Invalid Endpoint**
```
Azure OpenAI API request failed with status 404
```
Solution: Verify your `BASE_URL` and `DEPLOYMENT_NAME` environment variables.

### Debug Mode
Enable debug logging to see detailed API interactions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Environment Variable Checker
You can verify your environment variables are loaded correctly:

```python
import os
print("OpenAI API Key:", "✅ Set" if os.getenv('OPENAI_API_KEY') else "❌ Missing")
print("Base URL:", "✅ Set" if os.getenv('BASE_URL') else "❌ Missing")
print("Deployment Name:", "✅ Set" if os.getenv('DEPLOYMENT_NAME') else "❌ Missing")
print("API Version:", "✅ Set" if os.getenv('API_VERSION') else "❌ Missing")
``` 