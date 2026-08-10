import yaml
import logging
from typing import Dict, Any
from pathlib import Path

class ConfigurationError(Exception):
    """Custom exception for configuration related errors."""
    pass

class YAMLConfigManager:
    """
    YAML-based configuration management utility.
    """
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Cache for loaded configurations
        self._config_cache: Dict[str, Any] = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load YAML configuration file with error handling.
        
        :param config_name: Name of the configuration file (without .yaml extension)
        :return: Parsed YAML content
        :raises: ConfigurationError if loading fails
        """
        try:
            config_path = self.config_dir / f"{config_name}.yaml"
            
            # Check cache first
            if config_name in self._config_cache:
                return self._config_cache[config_name]
            
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self._config_cache[config_name] = config
                return config
                
        except FileNotFoundError:
            error_msg = f"Configuration file not found: {config_path}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
            
        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML format in file {config_path}: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
            
        except Exception as e:
            error_msg = f"Error loading configuration: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def save_config(self, config_data: Dict[str, Any], config_name: str) -> None:
        """
        Save configuration to YAML file.
        
        :param config_data: Configuration dictionary to save
        :param config_name: Name of the configuration file (without .yaml extension)
        :raises: ConfigurationError if saving fails
        """
        try:
            config_path = self.config_dir / f"{config_name}.yaml"
            
            with open(config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)
            
            # Update cache
            self._config_cache[config_name] = config_data
            
        except Exception as e:
            error_msg = f"Failed to save configuration: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def update_config(self, config_name: str, updates: Dict[str, Any]) -> None:
        """
        Partially update an existing YAML configuration.
        
        :param config_name: Name of the configuration file
        :param updates: Dictionary of updates to merge
        :raises: ConfigurationError if update fails
        """
        try:
            current_config = self.load_config(config_name)
            self._deep_update(current_config, updates)
            self.save_config(current_config, config_name)
            
        except Exception as e:
            error_msg = f"Failed to update configuration: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Usage example
if __name__ == "__main__":
    # Initialize the configuration manager
    config_manager = YAMLConfigManager()
    
    # Load an existing configuration
    config_data = config_manager.load_config("system_config")
    print(config_data)

    print(config_data["llm"]["model"])
    
    # Update the configuration
    config_manager.update_config("system_config", {"simulation": {"max_iterations": 10}})
    
    # Save a new configuration
    new_config = {
        "llm_api_endpoint": "https://api.openai.com",
        "llm_api_key": "your-api-key-here"
    }
    config_manager.save_config(new_config, "new_config")