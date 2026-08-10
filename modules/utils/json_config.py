import json
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import os
from pathlib import Path

class JSONConfigManager:
    """
    Comprehensive JSON configuration management utility.
    """
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """
        Load JSON file with error handling.
        
        :param file_path: Path to the JSON file
        :return: Parsed JSON content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Configuration file not found: {file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Invalid JSON format in file: {file_path}")
            return {}
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str) -> None:
        """
        Save data to a JSON file with pretty formatting.
        
        :param data: Dictionary to save
        :param file_path: Destination file path
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving JSON file: {e}")
    
    @staticmethod
    def update_json(file_path: str, updates: Dict[str, Any]) -> None:
        """
        Partially update an existing JSON file.
        
        :param file_path: Path to the JSON file
        :param updates: Dictionary of updates to merge
        """
        # Load existing data
        existing_data = JSONConfigManager.load_json(file_path)
        
        # Update with new data
        existing_data.update(updates)
        
        # Save updated data
        JSONConfigManager.save_json(existing_data, file_path)

class PromptTemplateManager:
    """
    Advanced prompt template management with variable interpolation.
    """
    def __init__(self, templates_dir: str = './prompts'):
        """
        Initialize prompt template manager.
        
        :param templates_dir: Directory containing prompt templates
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
    
    def load_prompt_template(self, template_name: str) -> str:
        """
        Load a prompt template from a JSON file.
        
        :param template_name: Name of the template file
        :return: Prompt template string
        """
        template_path = self.templates_dir / f"{template_name}.json"
        
        try:
            with open(template_path, 'r', encoding='utf-8') as file:
                template_data = json.load(file)
                return template_data.get('template', '')
        except FileNotFoundError:
            print(f"Prompt template not found: {template_name}")
            return ''
    
    def render_prompt(self, template_name: str, **kwargs) -> str:
        """
        Render a prompt template with variable interpolation.
        
        :param template_name: Name of the template
        :param kwargs: Variables to interpolate into the template
        :return: Rendered prompt
        """
        template = self.load_prompt_template(template_name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"Missing template variable: {e}")
            return template

# Example Configuration Class
@dataclass
class MetasurfaceDesignConfig:
    """
    Dataclass for storing metasurface design configuration.
    """
    llm_api_key: str
    design_library_path: str
    simulation_parameters: Dict[str, Any]
    
    @classmethod
    def from_json(cls, file_path: str):
        """
        Create configuration from JSON file.
        
        :param file_path: Path to configuration JSON
        :return: MetasurfaceDesignConfig instance
        """
        config_data = JSONConfigManager.load_json(file_path)
        return cls(**config_data)
    
    def save_to_json(self, file_path: str):
        """
        Save configuration to JSON file.
        
        :param file_path: Destination file path
        """
        JSONConfigManager.save_json(asdict(self), file_path)

# Example Usage
def main():
    # Configuration Management
    config_manager = JSONConfigManager()
    
    # Create a sample configuration
    config = {
        "llm_api_key": "your_api_key",
        "design_library_path": "./design_library.json",
        "simulation_parameters": {
            "max_iterations": 5,
            "frequency_range": [2.4, 2.5]
        }
    }
    
    # Save configuration
    config_manager.save_json(config, 'metasurface_config.json')
    
    # Load configuration
    loaded_config = config_manager.load_json('metasurface_config.json')
    
    # Prompt Template Management
    prompt_manager = PromptTemplateManager()
    
    # Example prompt template JSON (save in ./prompts/design_proposal.json)
    example_template = {
        "template": "Design a metasurface for {frequency} GHz with {polarization} polarization and gain > {gain} dB"
    }
    prompt_manager.save_json(example_template, './prompts/design_proposal.json')
    
    # Render prompt with variables
    rendered_prompt = prompt_manager.render_prompt(
        'design_proposal', 
        frequency=2.4, 
        polarization='circular', 
        gain=10
    )
    print("Rendered Prompt:", rendered_prompt)
    
    # Dataclass Configuration Example
    design_config = MetasurfaceDesignConfig(
        llm_api_key="test_key",
        design_library_path="./library.json",
        simulation_parameters={"iterations": 5}
    )
    
    # Save and load configuration
    design_config.save_to_json('design_config.json')
    loaded_design_config = MetasurfaceDesignConfig.from_json('design_config.json')

if __name__ == "__main__":
    main()