# import json
from modules.utils.yaml_config_manager import YAMLConfigManager
from pathlib import Path
from typing import Dict, Any

class PromptTemplateManager:
    """
    Advanced prompt template management with variable interpolation.
    """
    def __init__(self, templates_dir: str = './configs/prompt_templates'):
        """
        Initialize prompt template manager.
        
        :param templates_dir: Directory containing prompt templates
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
    
    def load_prompt_template(self, template_name: str) -> Dict[str, Any]:
        """
        Load a prompt template from a YAML file.
        
        :param template_name: Name of the template file
        :return: Prompt template string
        """
        template_path = self.templates_dir / f"{template_name}.yaml"
        
        try:
            return YAMLConfigManager.load_yaml(template_path)
        except FileNotFoundError:
            print(f"Prompt template not found: {template_name}")
            return None
    
    # def render_prompt(self, template_name: str, **kwargs) -> str:
    #     """
    #     Render a prompt template with variable interpolation.
        
    #     :param template_name: Name of the template
    #     :param kwargs: Variables to interpolate into the template
    #     :return: Rendered prompt
    #     """
    #     template = self.load_prompt_template(template_name)
    #     try:
    #         return template.format(**kwargs)
    #     except KeyError as e:
    #         print(f"Missing template variable: {e}")
    #         return template