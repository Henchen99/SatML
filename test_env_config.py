#!/usr/bin/env python3
"""
Test script to verify environment variable configuration for language models.
This tests the priority system: Environment Variables > Config > Defaults
"""

import os
import sys
import logging
from unittest.mock import patch

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_env_vars():
    """Check which environment variables are currently set."""
    logger.info("Checking environment variables...")
    
    env_vars = [
        'OPENAI_API_KEY',
        'BASE_URL', 
        'DEPLOYMENT_NAME',
        'API_VERSION',
        'LLAMA3_BASE_URL',
        'LLAMA3_API_KEY'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        status = "✅ Set" if value else "❌ Missing"
        logger.info(f"{var}: {status}")
    
    return any(os.getenv(var) for var in env_vars)

def test_azure_config_from_env():
    """Test Azure OpenAI configuration from environment variables."""
    logger.info("Testing Azure OpenAI with environment variables...")
    
    # Mock environment variables for testing
    test_env = {
        'OPENAI_API_KEY': 'test-azure-key',
        'BASE_URL': 'https://test-resource.openai.azure.com/',
        'DEPLOYMENT_NAME': 'gpt-4o-mini',
        'API_VERSION': '2023-07-01-preview'
    }
    
    with patch.dict(os.environ, test_env):
        try:
            from language_models.language_model_selection import LanguageModelFactory
            
            # Only non-sensitive config needed
            config = {
                "engine": "azure",
                "model": "gpt-4o-mini",
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            # This should work with just env vars
            model = LanguageModelFactory.create_model(config)
            logger.info("✅ Azure model created successfully from environment variables")
            
            # Verify the configuration was loaded correctly
            assert model.api_key == 'test-azure-key'
            assert model.base_url == 'https://test-resource.openai.azure.com/'
            assert model.deployment_name == 'gpt-4o-mini'
            assert model.api_version == '2023-07-01-preview'
            
            logger.info("✅ All Azure configuration values loaded correctly from environment")
            return True
            
        except Exception as e:
            logger.error(f"❌ Azure test failed: {e}")
            return False

def test_openai_config_from_env():
    """Test OpenAI configuration from environment variables."""
    logger.info("Testing OpenAI with environment variables...")
    
    # Mock environment variables for testing
    test_env = {
        'OPENAI_API_KEY': 'test-openai-key'
    }
    
    with patch.dict(os.environ, test_env):
        try:
            from language_models.language_model_selection import LanguageModelFactory
            
            # Only non-sensitive config needed
            config = {
                "engine": "openai",
                "model": "gpt-4o-mini",
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            # This should work with just env vars
            model = LanguageModelFactory.create_model(config)
            logger.info("✅ OpenAI model created successfully from environment variables")
            
            # Verify the configuration was loaded correctly
            assert model.api_key == 'test-openai-key'
            
            logger.info("✅ OpenAI configuration loaded correctly from environment")
            return True
            
        except Exception as e:
            logger.error(f"❌ OpenAI test failed: {e}")
            return False

def test_config_priority():
    """Test that environment variables take priority over config values."""
    logger.info("Testing configuration priority (env vars should override config)...")
    
    # Mock environment variable
    test_env = {
        'OPENAI_API_KEY': 'env-api-key'
    }
    
    with patch.dict(os.environ, test_env):
        try:
            from language_models.language_model_selection import LanguageModelFactory
            
            # Config has a different API key, but env var should take priority
            config = {
                "engine": "openai", 
                "model": "gpt-4o-mini",
                "api_key": "config-api-key"  # This should be ignored
            }
            
            model = LanguageModelFactory.create_model(config)
            
            # Environment variable should take priority
            assert model.api_key == 'env-api-key', f"Expected 'env-api-key', got '{model.api_key}'"
            
            logger.info("✅ Environment variables correctly take priority over config")
            return True
            
        except Exception as e:
            logger.error(f"❌ Priority test failed: {e}")
            return False

def test_missing_env_vars():
    """Test error handling when required environment variables are missing."""
    logger.info("Testing error handling for missing environment variables...")
    
    # Clear environment variables for testing
    with patch.dict(os.environ, {}, clear=True):
        try:
            from language_models.language_model_selection import LanguageModelFactory
            
            config = {
                "engine": "azure",
                "model": "gpt-4o-mini"
                # No config values and no env vars
            }
            
            # This should fail with helpful error message
            try:
                model = LanguageModelFactory.create_model(config)
                logger.error("❌ Should have failed with missing env vars")
                return False
            except ValueError as e:
                if "Missing required configuration" in str(e):
                    logger.info("✅ Correctly detected missing environment variables")
                    return True
                else:
                    logger.error(f"❌ Wrong error message: {e}")
                    return False
            
        except Exception as e:
            logger.error(f"❌ Missing env vars test failed: {e}")
            return False

def main():
    """Run all environment variable tests."""
    logger.info("Testing language models with environment variable configuration...")
    logger.info("="*60)
    
    # Check current environment
    has_env_vars = check_env_vars()
    if has_env_vars:
        logger.info("✅ Found some environment variables in current environment")
    else:
        logger.info("ℹ️  No environment variables found, using mock values for testing")
    
    logger.info("="*60)
    
    tests = [
        test_azure_config_from_env,
        test_openai_config_from_env,
        test_config_priority,
        test_missing_env_vars
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        logger.info("-" * 40)
        if test():
            passed += 1
    
    logger.info("="*60)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All environment variable tests passed!")
        logger.info("✅ The system correctly prioritizes environment variables over config")
        logger.info("✅ You can now safely store sensitive data in .env files")
        return 0
    else:
        logger.error(f"⚠️  {total - passed} tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main()) 