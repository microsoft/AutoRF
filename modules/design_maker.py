import logging
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass, asdict


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.surface_spec_def import *
from modules.literature_learner import DesignLitSummary
from modules.design_proposer import DesignProposal, DesignStrategy

from modules.design_info_def import MacroDesignDecision, DesignInfo
from modules.hfss_manager import HFSSManager, SimulationResults, SimulationStatus 
from modules.design_tuner import DesignTunerDecision, DesignTuner, TunerAction
from modules.design_reviewer import DesignReviewerFeedback, DesignReviewer

from modules.utils.custom_logger import setup_logger
from modules.utils.llm_interface import prompt_dict_to_string
from modules.utils.yaml_config_manager import YAMLConfigManager

from modules.rules_and_constrains import RulesAndConstrains
    

# class DesignTuningError(Exception):
#     """Custom exception for design tuning operations."""
#     pass

class OptimizationState:
    """Tracks the state of the optimization process."""
    def __init__(self):
        self.iteration: int = 0
        # self.best_performance: float = float('-inf')
        # self.best_design: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        # self.convergence_reached: bool = False

class DesignMaker:
    """
    Iteratively tunes metasurface design using HFSS simulation and LLM reasoning.
    """
    def __init__(self, llm_interface, llm_interface_o1, simulation_params: Dict[str, Any]):
        self.llm = llm_interface_o1
        self.llm_4o = llm_interface
        self.simulation_params = simulation_params

        self.logger = setup_logger()
        self.yaml_history_file = YAMLConfigManager(config_dir="./")

        self.hfss_manager = HFSSManager()

        self.design_rules = RulesAndConstrains()
        
        self.reviewer = DesignReviewer(llm_interface_o1)
        
        self.design_tuner = DesignTuner(llm_interface_o1)
        
        # Tuning configuration
        self.max_iterations = simulation_params.get('max_iterations', 3)
        # self.convergence_threshold = simulation_params.get('convergence_threshold', 0.95)
        
        # LLM output schema
        self.tuning_schema = {
            "parameter_adjustments": dict,
            "adjustment_reasoning": str,
            "confidence_score": float,
            "expected_improvement": float
        }

    def get_S_expr_list(self, target_spec: MetasurfaceSpec) -> List[str]:
        """
        Get the S-parameters expression for the target metasurface specifications.
        """
        expr_list = []
        # if target_spec.op_mode == SurfOpMode.REFLECT or target_spec.op_mode == SurfOpMode.TRANS_REFLECT:
        #     expr_list.append("S11")
        # if target_spec.op_mode == SurfOpMode.TRANS or target_spec.op_mode == SurfOpMode.TRANS_REFLECT:
        #     expr_list.append("S21")
        expr_list = ["S11", "S21"]

        return expr_list

    
    def make_design(self, template_info: DesignLitSummary, design_proposal: DesignProposal , target_spec: MetasurfaceSpec| AntennaSpec, design_id: int = 1) -> DesignInfo:
        """
        Perform iterative design optimization.
        
        param design_proposal: design tuning proposal, containing the template design and the tuning strategy for the mismatched spec
        param target_spec: Target metasurface specifications
        """
        self.logger.info("Starting design tuning process")
        state = OptimizationState()

        if isinstance(target_spec, MetasurfaceSpec):
            self.design_type = HardwareType.METASURFACE
        elif isinstance(target_spec, AntennaSpec):
            self.design_type = HardwareType.ANTENNA
        else:
            raise ValueError("Unknown design type!")

        ## 1. Prepare a HFSS design project for simulation
        self._prepare_design_project(design_proposal, design_id)

        ## 2. Initial Macro design reasoning
        # Decide Frequency change, Size scaling, material update, layer stacking etc.
        design_decision = self._macro_design_reasoning(template_info, design_proposal, target_spec)


        self._macro_design_update(design_decision)


        ## 3. Iterative design tuning (Micro reasoning with LLM)
        while state.iteration < self.max_iterations:
            self.logger.info(f"Starting micro iteration {state.iteration + 1}")
            
            # Run simulation
            try:

                simulation_results = self.hfss_manager.run_simulation(S_expr_list=self.get_S_expr_list(target_spec))

                # assert simulation_results.status == SimulationStatus.SUCCESS, "Simulation failed to run successfully."

                # performance_score = self._evaluate_performance(simulation_results, target_spec)
                tuner_decision, result_report = self.design_tuner.eval_and_tune(self.design_details, simulation_results, target_spec, state.history, template_info)

                
                # Update optimization state
                state.iteration += 1
                state.history.append({
                    "iteration": state.iteration,
                    "result_report": result_report,
                    "tuner_action": tuner_decision.tuner_action,
                    "tuner_decision": str(tuner_decision.model_dump()["reasoning_comment"]),
                    "design_variables": self.design_details.model_dump()["var_info_list"],
                    "design_sweeps": self.design_details.model_dump()["sweep_info_list"],
                    # "design_params": self.design_details.model_dump(),
                    # "simulation_results": simulation_results.model_dump(),
                })

                # Write down the history as a yaml file
                self.yaml_history_file.save_config(state.history, "design_tuning_history")
                
                # Check convergence
                if tuner_decision.tuner_action == TunerAction.SUCCESS_END:
                    self.logger.info("Convergence achieved, stopping optimization!!")
                    break

                if state.iteration == self.max_iterations:
                    self.logger.info("Maximum iterations reached, stopping optimization!")
                    if tuner_decision.tuner_action != TunerAction.SUCCESS_END:
                        self.logger.info("Not a success design solution!")
                else:
                    self.logger.info(f"Micro tuning update ...")
                    # Micro Tuning update here
                    self._micro_tuning_update(tuner_decision)
                
            except Exception as e:
                self.logger.warning(f"HFSS Simulation failed: {e}")
                raise e
                break
        
        # clean up the HFSS project
        self.hfss_manager.cleanup(close_projects=True)

        # Return current design?
        return state.history[-1] # Return the last design info as the final design info.
            
        

    def _prepare_design_project(self, design_proposal: DesignProposal, design_id: int) -> None:
        """Set up the HFSS project for the design tuning process."""

        template_name = design_proposal.template_name
        
        self.design_details = DesignInfo(design_id=design_id, 
                            design_filename= f"NewDesign_{design_id}.aedt", 
                            var_info_list=[], 
                            sweep_info_list=[],
                            object_info_list=[])
        
        # Create new design based on the template design project
        self.hfss_manager.setup_project_design(
            template_filename=template_name+"-template-design.aedt", 
            new_project_filename=self.design_details.design_filename,
            design_solution_type="Modal" if self.design_type is HardwareType.METASURFACE else "Terminal"
            )
        


    def _get_current_design_params(self) -> tuple[List, List]:
        """Get the current design parameters and parameter sweep info from the HFSS project."""

        # (1) Read design variables
        variables = self.hfss_manager.read_design_variables()
        self.logger.debug(f"Current Design variables:\n {variables}")

         # (3) Read old parameter sweep
        var_sweep_info_list, _ = self.hfss_manager.read_parameter_sweep()
        self.logger.debug(f"Current parameter sweep info:\n {var_sweep_info_list}")

        object_info_list = self.hfss_manager.read_design_objects_info()
        self.logger.debug(f"Current Design objects info:\n [{[obj.name for obj in object_info_list]}]")
        
        
        # store the design info
        self.design_details.var_info_list = variables
        self.design_details.sweep_info_list = var_sweep_info_list
        self.design_details.object_info_list = object_info_list

        # return variables, var_sweep_info_list


    def _macro_design_reasoning(self, template_info: DesignLitSummary, design_proposal: DesignProposal, target_spec: MetasurfaceSpec) -> MacroDesignDecision:
        pass
        # 0. Prepare general task information.
        messages = self._macro_design_prepare(template_info, design_proposal, target_spec)
        self.logger.debug(f"==== Initial/System prompt ==== \n {messages[0]["content"]} \n")

        # 1. Loop until review pass, make a decision and output the json format of the design edits.
        max_feedback_iteration = 2
        for i in range(max_feedback_iteration):
            # 1. Make initial design reasoning and planning based on the rules.
            messages, decision_obj = self._macro_design_planning(messages)

            if decision_obj is None:
                # 2. Parse response as structured data
                self.logger.info("No structured data obj available from response. Parsing the response with LLM now.")
                response = self._macro_design_output(messages)
            else:
                response = decision_obj

            assert isinstance(response, MacroDesignDecision), "Expecting MacroDesignDecision object."

            # Perform a review to make sure everything is correct
            feedback = self.reviewer.review_design_decision(messages) 
            # If verified, then break away, otherwise, loop back again.
            if feedback.passed == True:
                self.logger.info("Macro Design review Passed!")
                break
            elif i == max_feedback_iteration - 1:
                self.logger.info(f"Macro Design review Failed after {max_feedback_iteration} iterations!")
            else:
                self.logger.info(f"Macro Design review Failed, need to re-iterate {i+1} times!")
                messages.append({
                    "role": "user", "content": f"Your design edits above didn't pass the review. The reason is as fellow: {feedback.reasoning_comment}. Fix the issues mentioned and Try again!"
                })
                self.logger.debug(f"==== Add user message ==== \n {messages[-1]} \n")



        return response # Return the decision made by LLM



    def _macro_design_prepare(self, template_info: DesignLitSummary, design_proposal: DesignProposal, target_spec: MetasurfaceSpec) -> List[Dict[str, Any]]:
        
        # Get initial design parameters from the HFSS project
        self._get_current_design_params()

        messages = [
            {
                "role": "system",
                "content": """
You are a knowledgeable wireless (RF) hardware expert specializing in metasurfaces and antennas. You will be provided with hardware specifications for a target design you need to achieve, along with details of a template design as the starting point for modifications.
Your task is to determine the necessary macro design edits based on the template design. You'll also receive a design proposal that outlines the tuning strategy for any mismatches between the target and the template specs.
###
You can make the following edits:
1. **Set Analysis Frequency range for HFSS**: 
Determine the analysis frequency range, in the form of two float point numbers to specify the lower and upper limit of the range, using GHz as the unit.
2. **Edit Design Variables**: 
Propose updated HFSS design variable values to meet the expected hardware specifications, defining the nominal parameters of the design. To edit each variable, You need to provide information according to `VariablePropertyInfo` mentioned later. 
3. **Edit Parameter Sweeps**: 
Propose updated HFSS design variable parameter sweep values necessary for metasurface signal control (linear sweep with start, end, step values only, discrete value sweep with list_of_values is not supported yet.). To edit each variable, You need to provide information according `VariableSweepInfo` mentioned later.
NOTE do NOT increase the number of variations if we are using DUPLI_LAYERS as the design strategy, as it will cause too many variations in the design. 
4. **Edit substrate Material**: 
Propose updated substrate material for one or more named 3D box object in the design to meet the demanded material in the target specifications. To edit object material, you need to provide information according to `MaterialEditInfo(name:str, material:str)` python dataclass. 
5. **Delete and Add substrate layer**: 
Propose deleting or adding substrate layers. To edit object material, you need to provide information according to `ObjDeleteInfo(name:str)` and `BoxCreateInfo(name:str, position: List[str], size:List[float], material:str)` python dataclass. NOTE: DO NOT create any new variables! All positions and sizes MUST depend or reference existing design variables, to avoid 3D placement error, you can update the value of existing variables accordingly, for example, the thickness of new PCB FR4 substrate must be written as the variable names for original substrate's thickness `H` and `H` may be updated by frequency scaling.
6. **Create Design Variables**: (Only when adding metal sheet or copying surface layer) 
Propose new HFSS design variables to help define new design objects. For example, variable `surf_metal_gap` to define the gap between the surface and the added metal sheet, `inter_surf_gap` to define the gap/height between the original surface layer and the newly added copied surface layer. the  For each new variable, you need to provide information according to `VariablePropertyInfo` mentioned later.
7. **Create Metal Sheet**:
Add a new metal sheet object to the design to control the signal reflection. Typically, used to turning a transmissive surface to a reflective design, but do NOT use this to change a transflective metasurface. Note that port1 is at the upper of metasurface along Z axis, so the metal sheet should be below the surface -- lower Z-axis position `Position Z = surf_posZ - surf_metal_gap` -- so that S11 value is correct. To add a metal sheet, you need to provide information according to `MetalSheetCreateInfo` python dataclass or None if not needed. In the data class, the `position` should be a list of 3 float numbers, representing the X, Y, Z position of a corner of the metal sheet, and the `sizes` should be a list of 3 float numbers, representing the X, Y, Z size of the metal sheet along three axes. Metal sheet Z size is typically 0.1 mm.
8. **Copy Surface Layer**: 
Copy and paste a surface layer to create a new layer with the same pattern or independent pattern. For example, you can add an independent surface layer behind the current one to enlarge the design space for achieving the target specifications. To copy a surface layer, you need to provide information according to `SurfPatternCopyInfo()` python dataclass or None if not needed.
Unless otherwise mentioned in the design proposal, the new surface layer should be independent by default.
You should specify new layer's independent variables in `variables_toCopy` field only, and do NOT add new layer's variables through API for `Create Design Variables` above.
For example, for a passive surface with independent split ring pattern layer, we should copy the dimensional variables in the parameter sweep, such as radius "R", ring width "Rw". 
###
A metasurface design consists of the following components:
    1. **Substrate Layers Objects**: These are the dielectric layers that support the metallic patterns. 
    2. **Metallic Patterns Objects**: These are the metallic structures, typically coper, aluminum, or perfect conductor, that control the electromagnetic waves with specifically designed shapes, such as split ring, rectangular patches. 
    3. **Metallic Layer Objects (Ground Plane)**: These are the metallic layers, typically coper, aluminum, or perfect conductor, that provide ground plane for the design. 
    3. **Design Variables**: These are the parameters that define the physical dimensions and properties of various design objects, such as width, length, gap, and thickness of different objects. You can update independent variables to meet the target specifications.
    4. **Parameter Sweeps**: These are the ranges of values for design variables that allow for signal control, such as varying the radius of a split ring pattern can control the signal phase reflected by it.
###
You follow the following design rules:
                """
                + self.design_rules.rules_str() +
                "\nAfter calculating specific design parameter values, you also need to satisfy the following design constraints: \n" + 
                self.design_rules.constraints_str(),
            },
            {
                "role": "user",
                "content": "Task background information below:\n"
                + "The template design information is: \n"
                + str(template_info)
                + "\n###\n"
                + "The template HFSS design has the following 3D objects or structures, each having a description of position, size, and material: \n"
                + str(self.design_details.object_info_list)
                + "\n###\n"
                + "The template HFSS design has the following variables, each having a description of what it represents: \n"
                + str(self.design_details.var_info_list)
                + "\n###\n"
                + "The template HFSS design has the following variable parameter sweeps: "
                + str(self.design_details.sweep_info_list)
                + "\n###\n"
                + "The design proposal suggests the tuning strategy for the mismatched specs: \n"
                + str(design_proposal)
                + "\n###\n"
                + ("The target metasurface specifications are: \n" if self.design_type is HardwareType.METASURFACE else "The target antenna specifications are: \n")
                + str(target_spec),
            },
        ]

        return messages


    def _macro_design_planning(self, messages: Dict[str, Any]) -> tuple[str, MacroDesignDecision]:
        
        assert messages[-1]["role"] == "user", "Expecting user input as the last message."

        messages[-1]["content"] += """
###
Answer the following questions one by one:
1. **Overall Planning** To achieve the target specifications, how to describe the problem concisely? What are the necessary design edits you need to use? Explain in details why you chose those edits and how they should be used.
2. **Recall Background** What are the related background information you need to consider? For example, related equations of frequency scaling, material scaling, microstrip line width.
3. **Select variable or Object** Which variables, parameter sweeps, or objects you need to edit? Why do you choose them?
4. **Calculate values within constraints** What are the specific values of edits based on equations and the, what are the updated values for satisfying the constraints mentioned above? Provide a step-by-step calculation process for updating design variables and parameter sweeps. Use equations to calculate these values numerically and accurately for each updated variable and parameter sweep values.
5. **Summary** Summarize a list of all the edits you make with specific values of the edit. MUST explicitly mention variable and sweep values in the summary.
NOTE: Keep your reasoning and response concise -- within 1500 words for each, and do not repeat yourself!
"""
#         messages[-1]["content"] += """
# ###
# Answer a list of all the edits you make with specific values of the edit. MUST explicitly mention variable and sweep values in the summary.
# NOTE: Keep your reasoning and response concise -- within 1500 words for each, and do not repeat yourself!
# """

# """
# Perform the following steps and reply with your answers: 
# 1. **Overall Planning**: Read the background information provided to decide which kinds of edits you need to make. Explain in details why you chose those edits and how they should be used.
# 2. **Select variable or Object**: For each edit you choose to make, you may need to choose the specific variable, parameter sweep, or object to edit. For example, if you want to change the variables to modify operating frequencies, you need to choose the physical-dimension related variables to edit. If you want to change the substrate material, you need to choose the specific 3D box object in the HFSS design.
# 3. **Calculate values within constraints** Calculate or decide the specific values of edits while satisfying the constraints mentioned above. For example, provide a step-by-step calculation process for updating design variables and parameter sweeps. Use electromagnetic equations to calculate these values numerically and accurately for each updated variable and parameter sweep values. During calculation, you also need to consider the practical constraints, such as the realizable range and granularity of substrate thickness and design pattern. Typically, you can round up to the nearest values to meet the constraint. Otherwise, you can adjust multiple variables together to make trade-offs and satisfy the constraints. 
# 4. **Summary**: Summarize all the edits you made with specific values of the edit. MUST explicitly mention variable and sweep values in the summary.
# """

        self.logger.debug(f"==== Appended user message ==== \n {messages[-1]["content"]} \n")
        # Call LLM for macro design reasoning
        if self.llm.model_name == "deepseek-r1" or self.llm.model_name == "o1":
            response, data_obj = self.llm.get_json_response(messages, MacroDesignDecision)
            self.logger.info(f"LLM returned the following parsed json-based obj: {data_obj}\n")
        else:
            response = self.llm.get_string_response(messages)
            data_obj = None
            self.logger.debug(f"{self.llm.model_name} returned the response.\n")

        if response is None or len(response) < 10:
            self.logger.error(f"LLM response is empty or too short: {response}")
            raise ValueError("LLM response is empty or too short.")

        self.logger.debug(f"==== Macro Design Planning LLM response ==== \n {response} \n")

        messages.append({
            "role": "assistant",
            "content": response
        }) # Append the response.

        return messages, data_obj



    def _macro_design_output(self, messages: Dict[str, Any]) -> MacroDesignDecision:
        self.logger.info("Converting the LLM response to structured MacroDesignDecision output.")
        
        messages.append({
            "role": "user",
            "content": """
Perform the following steps and reply with your answers:
1. **Summary and Output**:  Summarize `assistant role` output above about the qualitative reasoning rationale or quantitative calculation process as the reasoning comment. Then, output all the chosen design edits and calculated final variable and sweep parameters, following the output schema.
"""
        })

        self.logger.debug(f"==== Add user message ==== \n {messages[-1]} \n")

        # Call LLM for macro design structured output
        response = self.llm_4o.get_structured_output(messages, MacroDesignDecision)

        messages.append({
            "role": "assistant",
            "content": response.model_dump_json()
        }) # Append the response.
        
        self.logger.debug(f"==== LLM response ==== \n {response} \n")

        return response




    def _macro_design_update(self, macro_decision: MacroDesignDecision) -> None:
        """
        Update the design based on the LLM response
        """
        pass

        ## 1. Set up simulation analysis setup
        assert len(macro_decision.analysis_freq_range) == 2, "Analysis frequency range must have two elements."
        analysis_params = {
            "frequency_range": macro_decision.analysis_freq_range,
            # Use HFSS manager default for other settings
        }
        self.hfss_manager.setup_analysis(analysis_params=analysis_params)
        
        ## 2. Update design variables
        if len(macro_decision.variable_updates) > 0:
            self.hfss_manager.update_design_variables(macro_decision.variable_updates)

        if macro_decision.variable_creates is not None and len(macro_decision.variable_creates) > 0:
            self.hfss_manager.create_design_variables(macro_decision.variable_creates)

        ## 3. Update parameter sweeps (delete old sweeps first, then add new one.)
        if len(macro_decision.sweep_updates) > 0:
            self.hfss_manager.delete_parameter_sweep(-1) # delete last one
            self.hfss_manager.create_parameter_sweep("Sweep1", macro_decision.sweep_updates)

        # 4. Update substrate material.
        if len(macro_decision.material_update) > 0:
            for material_info in macro_decision.material_update:
                self.hfss_manager.set_design_object_material(material_info.name, material_info.material)

        # 5. Delete old 3D objects
        if len(macro_decision.object3D_delete) > 0:
            for delete_info in macro_decision.object3D_delete:
                self.hfss_manager.delete_design_object(delete_info.name)

        # 5. Add new 3D objects
        if len(macro_decision.object3D_create) > 0:
            for create_info in macro_decision.object3D_create:
                self.hfss_manager.create_box_object(create_info.name, create_info.position, create_info.sizes, create_info.material) 

        # 6. Create metal sheet 
        if macro_decision.metal_create is not None:
            self.hfss_manager.create_metal_sheet(macro_decision.metal_create)
        
        if macro_decision.surf_copy is not None:
            self.hfss_manager.copy_surf_objects(macro_decision.surf_copy)

        # Update the local design info to the latest
        self._get_current_design_params()
        # Done.


    def _micro_tuning_update(self, micro_decision: DesignTunerDecision) -> None:
        """
        Update the design based on the LLM response
        """
        pass
        
        ## 1. Update design variables
        if len(micro_decision.variable_updates) > 0:
            self.hfss_manager.update_design_variables(micro_decision.variable_updates)

        ## 2. Update parameter sweeps (delete old sweeps first, then add new one.)
        if len(micro_decision.sweep_updates) > 0:
            self.hfss_manager.delete_parameter_sweep(-1) # delete last one
            self.hfss_manager.create_parameter_sweep("Sweep1", micro_decision.sweep_updates)


        # Update the local design info to the latest
        self._get_current_design_params()
        # Done.
    
    

## Testing the Design Maker
if __name__ == "__main__":
    # Load deisgn_proposal for testing from a pickle file
    import pickle

    with open('./new_designs/design_info_0.pkl', 'rb') as f:
        template_info, design_proposal, target_spec = pickle.load(f)
    # print(design_proposal)

    # Initialize the LLM interface
    # from modules.utils.llm_interface import OpenAILLMInterface
    # llm_interface = OpenAILLMInterface(model_name="gpt-4o")

    from modules.utils.llm_interface import OpenAILLMInterface
    llm_interface = OpenAILLMInterface(model_name="o1")
    # llm_interface = OpenAILLMInterface(model_name="gpt-4o")

    design_maker = DesignMaker(llm_interface=llm_interface, simulation_params={})

    # Test the design tuner
    design_maker.make_design(template_info, design_proposal, target_spec, design_id=1)

