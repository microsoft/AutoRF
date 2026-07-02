import logging
from typing import Dict, Any

# Import custom modules
from modules.utils.json_config import JSONConfigManager
from modules.utils.yaml_config_manager import YAMLConfigManager

from modules.utils.llm_interface import OpenAILLMInterface, GetLLMInterface
from modules.utils.prompt_manager import PromptTemplateManager

from modules.demand_translator import DemandTranslator
from modules.design_proposer import DesignProposer
from modules.literature_learner import DesignLitSummary
from modules.design_maker import DesignMaker

from modules.utils.custom_logger import setup_logger

class MetasurfaceDesignAutomation:
    """
    Orchestrates the entire metasurface design automation process.
    """
    def __init__(self, model_name = "o1", config_name: str = 'system_config'):
        """
        Initialize the design automation system.
        
        :param config_name: Path to the configuration file
        """
        self.logger = setup_logger()
        self.logger.info(f"Initializing design automation system...")

        # Load configuration
        self.config = YAMLConfigManager().load_config(config_name)

        # Initialize LLM interface
        # self.llm_interface = OpenAILLMInterface()

        if "deepseek" in model_name.lower():
            self.llm_interface = OpenAILLMInterface()
            self.llm_interface_o1 = GetLLMInterface(model_name="deepseek-r1", model_provider="Nvidia")
        else:
            self.llm_interface = OpenAILLMInterface()
            self.llm_interface_o1 = GetLLMInterface(model_name=model_name, model_provider="OpenAI")
        
        
        # Initialize modules
        self.demand_translator = DemandTranslator(self.llm_interface)

        self.design_proposer = DesignProposer(
            llm_interface=self.llm_interface_o1, 
        )
        self.design_maker = DesignMaker(
            llm_interface=self.llm_interface,
            llm_interface_o1=self.llm_interface_o1,
            simulation_params={}, 
        )

    def get_design_proposal(self, natural_language_demand: str) -> Dict[str, Any]:
        # Translate demand to specifications
        metasurface_spec = self.demand_translator.translate_demand(natural_language_demand)
        
        # Propose initial design
        template_info, design_proposal = self.design_proposer.propose_design(metasurface_spec)

        return template_info, design_proposal, metasurface_spec
    
    def get_final_design(self, template_info, design_proposal, metasurface_spec, design_id):
        self.logger.debug(f"Start making the design based on {design_proposal}.")
        try:
            if len(design_proposal.mismatched_spec) == 0:
                self.logger.info("Design proposal says perfect match found.")
                return template_info
            
            # Tune design
            final_design = self.design_maker.make_design(template_info, design_proposal, metasurface_spec, design_id=design_id)
            
            return final_design
        
        except Exception as e:
            self.logger.error(f"Metasurface design automation failed: {e}")
            raise e

    
    def automate_design(self, natural_language_demand: str, design_id: int = 1) -> Dict[str, Any]:
        """
        End-to-end metasurface design automation.
        
        :param natural_language_demand: User's high-level design requirements
        :return: Final optimized metasurface design
        """
        try:
            
            # Get design proposal and specifications
            template_info, design_proposal, metasurface_spec = self.get_design_proposal(natural_language_demand)

            if len(design_proposal.mismatched_spec) == 0:
                self.logger.info("Design proposal says perfect match found.")
                return template_info
            
            # Tune design
            final_design = self.design_maker.make_design(template_info, design_proposal, metasurface_spec, design_id=design_id)
            
            return final_design
        
        except Exception as e:
            self.logger.error(f"Metasurface design automation failed: {e}")
            raise e

def main():
    # Example usage
    design_system = MetasurfaceDesignAutomation()
    
    # Example user demand
    user_demands = YAMLConfigManager().load_config('user_demands')
    test_case_num = 1

    # Take a few examples from the user demands
    for i in range(test_case_num):
        user_demand = user_demands['my_examples'][i]['demand']
        
        # =========== specify your own user demand below  ==============
        user_demand = "Design a reflective passive metasurface for 120GHz band subTHz wireless coverage enhancement, using paper as substrate."
        # user_demand = "Design a reflective passive metasurface for 79GHz mmWave sensing, using FR4 as substrate."
        # user_demand = "Design a 2.4GHz patch antenna."
        # user_demand = "Design a programmable metasurface for 2.4GHz phase control of reflected signals for beamforming based wireless coverage extension."
        # user_demand = "Design a metasurface for 79GHz WiFi beamforming using phase control. It should be transmissive, low loss, and low-cost."

        # Automated design process
        optimized_design = design_system.automate_design(user_demand)
        print("Optimized Metasurface Design:")
        print(optimized_design)
    


if __name__ == "__main__":
    main()