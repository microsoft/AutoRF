if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils.custom_logger import setup_logger
from modules.utils.custom_logger import Colors
from modules.surface_spec_def import MetasurfaceSpec, AntennaSpec, HardwareType, HardwareTypeChoice


class DemandTranslationError(Exception):
    """Custom exception for demand translation errors."""
    pass

class DemandTranslator:
    """Translates natural language demands to structured metasurface specifications."""
    
    def __init__(self, llm_interface):
        self.llm = llm_interface

        self.logger = setup_logger()

        self.output_schema = MetasurfaceSpec

    def translate_demand(self, natural_language_demand: str) -> MetasurfaceSpec:
        """
        Convert natural language demand to structured specification.
        
        :param natural_language_demand: User's description of requirements
        :return: MetasurfaceSpec object
        :raises: DemandTranslationError if translation fails
        """
        try:
            self.logger.info("Starting demand translation")
            self.logger.debug(f"Input demand: {natural_language_demand}")

            # Add a check for metasurface or antenna design
            self.llm.logger.info("Check metasurface or antenna design first.")
            response = self.llm.call_llm({
                "system":"You are a helpful wireless (RF) hardware expert, working on metasurfaces and antennas.",
                "user": "Check the following user request to decide whether it is for metasurface. If metasurface, return true, for other design, such as antennas, return False. \n###\n" + natural_language_demand
            }, HardwareTypeChoice)

            # Check the response
            if response.hardware_type == HardwareType.ANTENNA:
                self.logger.info("Starting the Antenna design process.")
            elif response.hardware_type == HardwareType.METASURFACE:
                self.logger.info("Starting the Metasurface design process.")

            

            prompt = {
                "system": 
                    "You are a helpful wireless (RF) hardware expert, working on metasurfaces and antennas. You translate users' application demand to metasurface hardware specifications."
                    "Think carefully, step by step.\n"
                    "(1) First, note down your reasoning process as a concise comment for why each spec is chosen. (2) Then fill in the corresponding fields in the output schema. If the user requested any specification specifically, you MUST use that. Otherwise, reason about user intent based on user input and domain knowledge, but must not make any unfounded guess. "
                    "If any field is not mentioned, use default value or put -1 or 'None' to show there is not constraint on that specifications.\n"
                    "###\nFollow the following general rules and examples:\n" 
                    "0. Always find the simplest specification that can meet users' demands with minimal design complexity and difficulty."
                    "1. For operating frequency bands, follow users' input if possible, output center frequency points and frequency ranges in GHz unit. If no direct information available, infer based on other information, such as 5GHz for WiFi means 5.5GHz center frequency."
                    "2. If user demand specifies the frequency range or bandwidth, calculate based the given information. Otherwise, if no information given, use 0.01 as the default bandwidth ratio and set the frequency range as the center frequency. For example, if the center frequency is 5GHz, the frequency range is [5, 5] GHz. Only use at most one digit after the decimal point." # Skip the bandwidth issue for now.
                    # "2. Can infer the operation mode based on metasurface feature or functionalities, e.g., transmissive for through-wall, reflective for on, and absorptive for stealth."
                    "3. Choose the polarization type as `AllAllowed` by default, unless user or the specific application requires otherwise. "
                    "4. The default min_power_efficiency_dB is -10dB for all designs, unless otherwise specified. This is to ensure that the design is not too lossy. If the user requires low loss, use -4dB instead, or use the specified value from the user."
                    "5. For the signal control mode, you can reason based on other info, e.g., programmable if `reconfigurable or adapting to dynamics` is needed, passive or fixed if `zero-power consumption, no control wiring,  low-cost` is mentioned. Use Passive by default."
                    "6. For the signal control type, use phase by default, unless otherwise instructed."
                    "7. For the deployment material, select ANY material if no information provided. You can infer material based on implicit information, for example, use `FR4` when mention standard PCB fabrication, use `paper` and `passive type` when need `low cost`, use `Rogers` when need `low power loss`, use FPC when need `flexible surface`. For other fields in `deployment_spec`, default value for each field is `max_num_of_layers = 10`, `max_thickness = 100`, `max_dollar = 10`.",
                "user": natural_language_demand
            }
            
            # Call LLM for translation
            response = self.llm.call_llm(prompt, self.output_schema if response.hardware_type == HardwareType.METASURFACE else AntennaSpec)
            self.logger.debug(f"LLM response: {response}")

            surface_spec = response
            
            # Validate frequency range
            # freq_range = (surface_spec.freq_range_low, surface_spec.freq_range_high)
            # if freq_range[0] >= freq_range[1]:
            #     raise DemandTranslationError("Invalid frequency range: min >= max")
            
            # Validate polarization
            # try:
            #     polarization = surface_spec.polarization
            # except KeyError:
            #     raise DemandTranslationError(f"Invalid polarization type: {response['polarization']}")
            
            
            
            self.logger.info("Demand translation completed successfully")
            self.logger.debug(f"Generated specification: {surface_spec}")
            
            return surface_spec
            
        except KeyError as e:
            error_msg = f"Missing required field in LLM response: {e}"
            self.logger.error(error_msg)
            raise DemandTranslationError(error_msg)
            
        except ValueError as e:
            error_msg = f"Invalid value in specification: {e}"
            self.logger.error(error_msg)
            raise DemandTranslationError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error during demand translation: {e}"
            self.logger.error(error_msg)
            raise DemandTranslationError(error_msg)
        

# Usage example

if __name__ == "__main__":
    # Initialize the LLM interface
    from modules.utils.llm_interface import OpenAILLMInterface
    from modules.utils.yaml_config_manager import YAMLConfigManager

    llm_interface = OpenAILLMInterface()
    
    
    # Initialize the demand translator
    demand_translator = DemandTranslator(llm_interface)
    
    # Translate demand to specifications
    try:
        # Example user demand
        # user_demand = "Design a metasurface for 5G mmWave celluar coverage with linear polarization."
        user_demands = YAMLConfigManager().load_config('user_demands')
        test_case_num = 1

        # Take a few examples from the user demands
        for i in range(test_case_num):
            user_demand = user_demands['my_examples'][i]['demand']
        
            metasurface_spec = demand_translator.translate_demand(user_demand)
            print("Metasurface Specifications:")
            print(Colors.CYAN + str(metasurface_spec))
            print(Colors.RESET)
    
    except DemandTranslationError as e:
        print(f"Demand translation failed: {e}")