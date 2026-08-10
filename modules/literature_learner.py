if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, asdict
from pydantic import BaseModel
import pickle


from modules.surface_spec_def import MetasurfaceSpec, AntennaSpec
from modules.utils.custom_logger import setup_logger

from modules.utils.pdf_extraction import PDFExtractor


# @dataclass
class DesignLitSummary(BaseModel):
    """
    Structured summary of metasurface design based on literature.
    It includes a concise summary of the original application, design description, the structured specification for the hardware.
    """
    nickname: str
    application_scenario: str
    design_description: str
    design_params: str
    # structured metasurface specification
    hardware_spec: MetasurfaceSpec | AntennaSpec 


class LitratureLearner:
    """
    Learn the structured specification for metasurface design from literature.
    """
    def __init__(self, llm_interface):
        self.llm = llm_interface

        self.logger = setup_logger()


        self.output_schema = DesignLitSummary
        
    
    def summarize_lit(self, system_name, literature_path) -> DesignLitSummary:
        """
        Take in the literature, either as a file path or a URL link, and summarize the structured specification for metasurface design.
        
        :param litrature_path: Path to the literature file or URL link to the literature
        :return: DesignLitSummary object
        """
        try:
            self.logger.info("Starting liturature summary")
            self.logger.debug(f"Input demand: {literature_path}")
            
            if literature_path.startswith("http"):
                # download the literature file to local folder
                lit_file_path = "../design_library/paper_pdf/"
                # download the file and save it to the local folder
                pass
                # extract the text from the file
                raise NotImplementedError("Downloading literature from URL is not implemented yet.")
        
            else:
                # read the file content
                self.pdf_content = PDFExtractor(literature_path).extract_content()

            prompt = {
                "system": 
                    "You are a helpful wireless (RF) hardware expert, working on metasurfaces and antennas. You read research papers then summarize metasurface hardware design details."
                    ,
                "assistant": 
                    "Follow the following general rules and examples:" 
                    "Think carefully, step by step. Follow the schema to fill in the structured output."
                    # give the nickname of the design, typically use the system name by the author or you can make up a name based on hardware characteristics. 
                    "First, summarize the application scenario in one or two sentences concisely. Then summarize the hardware design, focusing on the metasurface unit cell element design or antenna 3D structure. This part should not mention applications, but cover hardware functionalities, the surface pattern and structure description for reproducing the design, for example, `two layers of metallic patChes, interconnected with diodes. each layer has 5 mm thick substrate`. Then, record all hardware design parameters mentioned in the paper, such as `element spacing is 10~mm`, `pattern width is 6~mm and substrate thickness is 4mm` in the format of dictionary. Lastly, carefully fill in the structured specification for the hardware. "
                    "Note the following: 1. unless specifically mentioned support circular or cross-polarization, the metasurface design supports only linear polarization. 2. for TransRing design specially, the operating frequency is 60GHz, different the user input."
                    ,
                "user": f"System nickname is {system_name}.\n" + self.pdf_content["metadata"]["abstract"] + self.pdf_content["main_text"] 
                # Add more specific instructions based on the system name
                        + (" Note that we only use the quarter wave plate design for this template, focus on quarter wave plate design details only." if "QWP" in system_name else "")
            }
            
            # Call LLM for translation
            response = self.llm.call_llm(prompt, self.output_schema)
            self.logger.debug(f"LLM response: {response}")

            design_summary = response

            
            self.logger.info("Literature completed successfully")
            # self.logger.debug(f"Generated specification: {design_summary}")
            
            return design_summary
            
            
        except Exception as e:
            error_msg = f"Unexpected error during literature learning: {e}"
            self.logger.error(error_msg)
            raise e


def review_and_correct():
    """
    Review the structured design summary and correct any mistakes.
    """
    
    nickname = "TransRing"
    
    with open(f"./design_library/design_summary/{nickname}.pkl", 'rb') as file:
        design_summary = pickle.load(file)
    
    print(design_summary)

    # Correct the design summary
    pass

    ## RFocus design correction
    # design_summary.
    ## Hydra design correction
    # Does not support 5GHz band
    # design_summary.hardware_spec.second_freq_band = 0
    # design_summary.hardware_spec.second_freq_range = [0, 0]
    
    ## TransRing design correction
    # Operating frequency is 60GHz
    # design_summary.hardware_spec.polarization=PolarizationType.CROSS
    design_summary.hardware_spec.min_power_efficiency_dB = -10

    with open(f"./design_library/design_summary/{nickname}.pkl", 'wb') as file:
        pickle.dump(design_summary, file)


def generate_all_lit_summaries():
    """
    Generate structured design summaries for all literature.
    """
        # Initialize the LLM interface
    from modules.utils.llm_interface import GetLLMInterface
    from modules.utils.yaml_config_manager import YAMLConfigManager

    llm_interface = GetLLMInterface(model_name="o1", model_provider="OpenAI")
    
    
    # Example usage
    lit_learner = LitratureLearner(llm_interface)

    folder_path = "./design_library/paper_pdf/"

    nicknames = ["AutoMS", "RF-Mediator", "PMSat", "RFocus", "Hydra", "TransRing", "LLAMA-QWP"]
    paper_list = ["automs-final.pdf", "rf-mediator.pdf", "pmsat.pdf", "rfocus.pdf", "hydra.pdf", "MSITM.pdf", "LLAMA-QWP.pdf"]
    # nicknames = ["TransRing"]
    # paper_list = ["MSITM.pdf"]


    for id, paper in enumerate(paper_list):
        design_summary = lit_learner.summarize_lit(nicknames[id], folder_path + paper)

        # dump the design summary to the design library folder as a object
        with open(f"./design_library/design_summary/{nicknames[id]}.pkl", 'wb') as file:
            pickle.dump(design_summary, file)

        # # save the design summary to the design library folder as a yaml file
        # YAMLConfigManager().save_config(asdict(design_summary), "{design_summary.nickname}")
        

        # print(design_summary)



if __name__ == "__main__":
    # generate_all_lit_summaries()

    review_and_correct()
