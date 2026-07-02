import logging
from typing import Dict, Any, List

import yaml, pickle
from pathlib import Path

from pydantic import BaseModel
from dataclasses import dataclass
from enum import Enum


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.surface_spec_def import MetasurfaceSpec, AntennaSpec, HardwareType
from modules.literature_learner import DesignLitSummary
from modules.circuit_model_simulator import CircuitModelSimulator
from modules.design_info_def import DesignProposal, DesignStrategy

from modules.utils.custom_logger import setup_logger



class DesignProposalError(Exception):
    """Custom exception for design proposal operations."""
    pass

class DesignProposer:
    """Proposes initial metasurface designs based on specifications."""
    
    def __init__(self, llm_interface, design_library_path: str = "./design_library/design_summary/"):
        self.llm = llm_interface
        self.design_library_path = Path(design_library_path)
        self.logger = setup_logger()
        
        # Initialize design library cache
        self.design_library = self._load_design_library()

        # Initialize the circuit model simulator
        self.circuit_model_simulator = CircuitModelSimulator(llm_interface)
        
        # Define output schema for LLM
        self.output_schema = DesignProposal
    
    def _load_design_library(self) -> Dict[str, Any]:
        """
        Load and validate the design template library.
        
        :return: Dictionary containing design templates
        :raises: DesignLibraryError if library loading fails
        """
        try:
            # Load design library
            self.logger.info("Loading design library from file %s", self.design_library_path)
            # with open(self.design_library_path, 'r') as f:
            #     library = yaml.safe_load(f)

            # Read all pkl files in the folder
            library = {}
            for file in self.design_library_path.glob("*.pkl"):
                with open(file, 'rb') as f:
                    design_name = file.stem # get the file name without extension
                    design_summary = pickle.load(f) 
                    library[design_name] = design_summary

            
            # Validate library structure
            # self._validate_library_structure(library)
            
            self.logger.info("Design library loaded successfully")
            return library

            
        except Exception as e:
            error_msg = f"Error loading design library: {e}"
            self.logger.error(error_msg)
            raise e
    
    def get_design_experiences(self) -> str:
        """
        Get the design experiences that guide the design proposer
        """

        design_experiences = [
            "1. When choosing among a few close match templates, you need to choose the one that simplifies later design tuning. ",
            "2. Fewer mismatched spec often means easier tuning, unless other experience says otherwise.",
            "3. Certain mismatched spec, like frequency tuning, substrate material change, is easier to tune."
            "4. Signal control mode, programmable or passive non-reconfigurable, are infeasible to switch by later tuning."
            # "5. Impedance matching metasurface, RF-mediator should not be used for amplitude control demand.", # Special case, ignore for testing.
            ## For new layer based designs
            "6. By adding a metal sheet for reflection, you can turn a transmissive impedance-control or continuous amplitude control surface to reflective and this also helps provide a large phase control range by utilizing resonance between metasurface and the metal sheet. The thickness between the surface and the metal sheet, and the surface admittance of the new surface layer are the new design variables. NOTE that this design strategy can be combined with other strategies like frequency scaling to adjust operating frequency as well, or material change",
            "7. By adding a extra copied surface layer, you can potentially improve the performance of a transmissive metasurface, such as higher transmissive power. The thickness between the surface and the new layer, and the surface admittance of the two surface layers are the new design variables.",
            # Mostly to limit design space, avoid very long simulation. -- limit to one new layer.
            "8. For now, we only consider adding one new layer -- either one new metal sheet or one new metasurface layer. We do not consider adding multiple new layers or reorganizing the existing layers. Do NOT add new metal or new layer unnecessarily such as turning a transflective (both trans and reflective) surface to reflective only is not useful or necessary. ",
        ]

        return "\n".join(design_experiences)

    
    def propose_design(self, spec: MetasurfaceSpec) -> Dict[str, Any]:
        """
        Find suitable initial design template and parameters.
        
        :param spec: Metasurface hardware specification
        :return: Initial design template file name
        :raises: DesignProposalError if proposal generation fails
        """
        try:
            self.logger.info("Starting design proposal generation")
            # self.logger.debug(f"Input specification: {spec}")
            
            # Get hardware type from specification
            hardware_type = HardwareType.METASURFACE if isinstance(spec, MetasurfaceSpec) else HardwareType.ANTENNA
            self.logger.debug(f"Proposing design for hardware type: {hardware_type}")

            # Create input for LLM
            input = {
                "specification": spec,
                "available_templates": self._get_template_summaries(hardware_type),
            }
            
            # Generate prompt for template selection
            # prompt = (
            #     f"Based on the specification {context['specification']}, "
            #     f"select the most suitable template from: {context['available_templates']}. "
            #     f"Consider these constraints: {context['design_constraints']}. "
            #     "Provide template ID, initial parameters, and scaling factors."
            # )

            prompt = {
                # 1. System provides the context and instructions
                "system": 
                "You are a helpful wireless (RF) hardware expert, working on metasurfaces and antennas. Based on the demanded hardware specification input, you need to choose the suitable design template. If it is not a perfect match, based on the mismatched specifications, you need to propose a design strategy for later design stages.\n" +
                # "You need to think carefully, step by step."
                f"You should follow the following design experiences:\n {self.get_design_experiences()} \n ### \n"
                # "You follow to following quantitative reasoning to find the template with HIGHEST SCORE to find the match with minimal difference: "
                # "Based on the difficulty of tuning that specification if mismatched, we assign scores to each matched specification"
                # " -- 1 pt for first and second operating frequency band, 2 pt for bandwidth ratio, 2 pt for operation mode, 2 pt for polarization,."
                # "First calculate the scores the for the five most possible matches, give the calculation process in the reasoning comment."
                # "Then choose the highest score one and output following the output schema."
                ,
                # 2. Assistant provides the available templates
                "user": 
                "###"+
                "Here are the metasurface templates available in the library:\n" +input["available_templates"]+
                "###"
                "The demanded hardware specification is "+ str(input["specification"]) +
                # """
                # STEPS:
                # First, MUST check the available template designs in the library provided below and find the top 2 or 3 options. Then, think about which template is the most suitable to achieve the given specification -- prefer a perfect match, otherwise a close match with some mismatched spec. After choosing a close match, you need to record the mismatched specifications and choose a design strategy for later tuning. If no suitable design tuning strategy is available, you can propose a new design strategy. In the worst case that the specification is impossible to meet, you need to choose the impossible design strategy. For any choice you make, you need to provide the rationale in the strategy comment.
                # """
f"""
###
You have the following design strategies to choose from:
{DesignStrategy.print_list()}
###
Answer the following questions:
1. **Identify Candidates**: 
    Which templates are the top 2 or 3 possible options, based their design summary and the demanded specification?
2. **Find simplest template and design strategies**: 
    Considering all available design strategies above, choose the best template and one or multiple corresponding design strategies to achieve demanded specifications. For example, if the design is a perfect match, choose it and PERFECT_MATCH strategy. If a metasurface template has frequency and substrate spec mismatched, choose FREQ_SCALE and MATERIAL_CHANGE. If no suitable design strategy is available, propose a new design strategy. If the specification is impossible to meet, choose the impossible design strategy. 
3. **Construct the Circuit Model**: 
    If you choose to add a metal sheet or a new surface layer, what is the high-level structure or blocks of the cascaded circuit model? For example,  `Air - Metasurface - Air Spacer - Metal sheet` if adding a metallic sheet to turn a transmissive surface to reflective. Also list the new design variables that can be used in the circuit model simulation. For example, `thickness between surface and metal sheet`, `surface admittance of the new surface layer`. If no new layer needs to be added, make sure return empty string for `circuit_model_blocks`.
NOTE: Keep your reasoning and response concise, and do not repeat yourself!
"""
    # And what are the new design variables that we can use? For example, `thickness between surface and metal sheet`, `surface admittance of the new surface layer`

            }

            self.logger.debug(f"==== Generated prompt ==== \n {prompt} \n")
            
            # Get LLM proposal
            response = self.llm.call_llm(prompt, self.output_schema)
            self.logger.debug(f"==== LLM response ==== \n {response} \n")
            
            
            design_proposal = response
            self.logger.info("Design proposal generated successfully")
            # self.logger.debug(f"Generated proposal: {design_proposal}")

            assert design_proposal is not None, "Design proposal is None, LLM response failed."

            circuit_blocks = design_proposal.circuit_model_blocks
            if circuit_blocks and circuit_blocks != "":
                # Simulate the proposed circuit model
                circuit_sim_guidance = self.circuit_model_simulator.get_guidance(spec, design_proposal)

                # Append simulation guidance to the design proposal
                design_proposal.strategy_comment += "\nAccording to circuit model simulation, we know " + circuit_sim_guidance
            
            return self.design_library[design_proposal.template_name], design_proposal
            
        except KeyError as e:
            error_msg = f"Invalid template reference: {e}"
            self.logger.error(error_msg)
            raise DesignProposalError(error_msg)
            
        except Exception as e:
            error_msg = f"Error generating design proposal: {e}"
            self.logger.error(error_msg)
            raise DesignProposalError(error_msg)
    
    def _get_template_summaries(self, hardware_type: HardwareType = HardwareType.METASURFACE) -> str:
        """Get a text summary of all available templates."""
        summary_all = ""


        for name, design_summary in self.design_library.items():
            if isinstance(design_summary.hardware_spec, MetasurfaceSpec) and hardware_type == HardwareType.METASURFACE:
                if "hydra" in name.lower() or "pmsat" in name.lower(): # or "mediator" in name.lower()
                    self.logger.info(f"Design template {name} is skipped.")
                    continue
                summary_all += f"- Design name: {name}, Information: {str(design_summary)}\n"
            elif isinstance(design_summary.hardware_spec, AntennaSpec) and hardware_type == HardwareType.ANTENNA:
                summary_all += f"- Design name: {name}, Information: {str(design_summary)}\n"
            else:
                self.logger.debug(f"Design {name} is not a {hardware_type.name} design, skipped.")
                continue

        return summary_all

    


# Usage example
NEW_DESIGN_FOLDER = "./new_designs"

if __name__ == "__main__":

    print(DesignProposal.model_json_schema())
    exit() 

    # Initialize the LLM interface
    from modules.utils.llm_interface import OpenAILLMInterface
    from modules.utils.yaml_config_manager import YAMLConfigManager
    from modules.demand_translator import DemandTranslator
    from modules.utils.custom_logger import Colors


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

            # Initialize design proposer
            design_proposer = DesignProposer(
                llm_interface=llm_interface, 
            )

            # Propose initial design
            template_info, design_proposal = design_proposer.propose_design(metasurface_spec)

            # store new design info as "design_info_{id}.pkl" in the NEW_DESIGN_FOLDER
            # first check if the folder exists
            Path(NEW_DESIGN_FOLDER).mkdir(parents=True, exist_ok=True)
            
            # # check the last id
            # id = 0
            # for file in Path(NEW_DESIGN_FOLDER).glob("design_info_*.pkl"):
            #     id = max(id, int(file.stem.split('_')[-1]))
            # # increment the id for the new design
            # id += 1
            
            id = i
            print("Dump the design proposal as pickle file for later testing.")
            with open(Path(NEW_DESIGN_FOLDER) / f"design_info_{id}.pkl", 'wb') as f:
                pickle.dump((template_info, design_proposal, metasurface_spec), f)
            
    
    except Exception as e:
        print("Design Proposer failed: ", e)
        raise e