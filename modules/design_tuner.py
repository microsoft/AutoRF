if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils.custom_logger import setup_logger
from modules.utils.custom_logger import Colors
from modules.utils.llm_interface import prompt_dict_to_string


from modules.surface_spec_def import MetasurfaceSpec, AntennaSpec
from modules.design_info_def import MacroDesignDecision, DesignInfo
from modules.hfss_manager import SimulationResults, VariablePropertyInfo, VariableSweepInfo, SimulationStatus
from modules.design_evaluator import SimResultsEvaluator
from modules.rules_and_constrains import RulesAndConstrains
from modules.literature_learner import DesignLitSummary

from pydantic import BaseModel
from enum import Enum
from typing import List




class TunerAction(Enum):
    SUCCESS_END = "Successful completion"
    FAILURE_END = "Failed to complete, no solution found" # Reverting back to macro design reasoning.
    TUNE_CONTINUE = "Continue next tuning iteration with updated design parameters"


class DesignTunerDecision(BaseModel):
    """Decision of the design tuner with LLM interface"""
    # 1. Summary of the HFSS simulation results, with semantic information. % Skipped, moved to the design evaluator.
    # results_summary: str
    # 1. **Simulation Result Summary**: Summarize the HFSS simulation results as quantitative descriptions of key metrics, especially focus on the metrics that are relevant to the target specifications. For example, if the target specifications mention being reflective on a specific frequency range, you should summarize the reflection coefficient on that range. More importantly, you should summarize potentials to achieve the target specifications. For example, if the target specifications mention phase shift control of around 360 degrees, but the evaluation results show only a phase shift of 180 degrees on the target frequency, but 360 degrees on a adjacent different frequency band, you should summarize that the design can be tunned to achieve that spec.

    # 2. Reasoning comment and overall action to take.
    reasoning_comment: str
    tuner_action: TunerAction
    
    # 3. Calculation process for the decision, with reasoning and electromagnetic equations.
    calculation_process: str
    # 4. Updated design variables and parameter sweeps for the next HFSS design tuning.
    variable_updates: List[VariablePropertyInfo]
    sweep_updates: List[VariableSweepInfo]


class DesignTuner:
    """Evaluate the results of the design and decide further tuning step with LLM interface"""
    
    def __init__(self, llm_interface):
        self.llm = llm_interface

        self.logger = setup_logger()
        self.tuning_rules = RulesAndConstrains()

        self.output_schema = DesignTunerDecision
        
        
    def eval_and_tune(self, current_design: DesignInfo, results: SimulationResults, target_spec: MetasurfaceSpec| AntennaSpec, tuning_history: list[dict], template_info: DesignLitSummary) -> DesignTunerDecision:
        """
        Evaluate the results of the design with LLM interface and return the tuning decision.
        
        """
        try:
            self.logger.info("Starting design tuner decision process")

            ## format the history of previous tuning decisions and results
            # each entry in the list is as follow
            # {
            #         "iteration": state.iteration,
            #         "result_report": result_report,
            #         "tuner_decision": tuner_decision.model_dump()["tuner_action"],
            #         "design_variables": self.design_details.model_dump()["var_info_list"],
            #         "design_sweeps": self.design_details.model_dump()["sweep_info_list"],
            # }
            tuning_history_str = ""
            for i, history_entry in enumerate(tuning_history):
                # if i == 0:
                    # tuning_history_str += "Iteration 0:\n"
                # tuning_history_str += f"Result report: {history_entry['result_report']}\n"
                tuning_history_str += f"Iteration {i+1}:\n"
                tuning_history_str += f"Tuner decision: {history_entry['tuner_decision']}\n"
            
            if len(tuning_history) == 0:
                tuning_history_str += "No previous tuning history."

            ## Handling different Simulation results status here.
            if results.status == SimulationStatus.SUCCESS:
                # 1. Success. Evaluate current design results
                self.result_evaluator = SimResultsEvaluator(target_spec, results)
                result_report = self.result_evaluator.parse_results()
            elif results.status == SimulationStatus.HFSS_ERROR:
                # 2. HFSS Error. use the HFSS error messages as the evaluation results
                result_report = "No result available as HFSS stopped due to design validation error. Error Messages below:\n" + results.hfss_message

            ## Prepare the prompt for LLM
            prompt = {
                "system": f"""
You are a knowledgeable wireless (RF) hardware expert specializing in metasurfaces and antennas. You will be provided with hardware specifications for a target design you need to achieve, along with the HFSS simulation result summary report of the current design.
Your task is to evaluate the simulation results against the target specifications and Answer the following questions in the output schema:
###
1. **Evaluate and Decision**: Has the target specifications been achieved? Based on the simulation result summary, evaluate the design performance. If all target specifications have been achieved, consider a success and return with TunerAction.SUCCESS_END. Even if a better frequency is suggested, when all specifications are achieved, stop tuning by return TunerAction.SUCCESS_END. If not all specifications achieved, continue tuning with updated variables and sweeps. (NOTE: Ignore the bandwidth requirement for antennas for now.)
2. **Tuning parameters within constraints**: If you decide to continue tuning for the last question, What updated design variables and parameter sweeps for the next HFSS design tuning should we use? Note that do NOT change the sweep of circuit component variables, such as resistance, capacitance, as they are pre-determined. For now, we only support tune a smaller step size for capacitance sweep. Also do not remove existing design variables in the sweep, only update their sweep range and step. Provide a detailed calculation or reasoning process for your decision, preferably with electromagnetic equations, meanwhile satisfying all the constraints mentioned below. Make sure that you only provide necessary design variables and parameter sweeps that needs edits, not the entire design or any unnecessary edits. 
###
MUST follow these General Rules:
{self.tuning_rules.tuning_rules_str()}
And the following Constraints: 
{self.tuning_rules.constraints_str()}
                """,
# Steps:
#     1. **Reasoning and Decision**: Based on the simulation result summary, first provide a reasoning comment about the design, describing whether each of the target specifications has been achieved. If all target specifications have been achieved, consider a success and return with TunerAction.SUCCESS_END. If not all achieved, continue tuning with updated variables and sweeps. 
#     2. **Calculation of tuning parameters within constraints**: If you decide to continue tuning, provide updated design variables and parameter sweeps for the next HFSS design tuning. Provide a detailed calculation or reasoning process for your decision, preferably with electromagnetic equations, meanwhile satisfying all the constraints mentioned below. Make sure that you only provide necessary design variables and parameter sweeps that needs edits, not the entire design or any unnecessary edits. 
#                 Note: (Not Used any more.)
# Only for metasurface designs, you can use parameter sweep_updates, while for antenna design tuning, you should not use sweep_updates and only update the design variables!
                "user": 
                "Task background information below:"+
                "\n###\n" + 
                ("The target metasurface specifications are: \n" if isinstance(target_spec, MetasurfaceSpec) else "The target antenna specifications are: \n")+
                str(target_spec) +
                "\n###\n" +
                "The current design is modified from the following template design: \n" + str(template_info) +
                "\n###\n" + 
                "The detailed information of the current design: \n" + 
                "Design variables: " + str(current_design.var_info_list) + "\nDesign parameter sweeps:" + str(current_design.sweep_info_list) +
                "\n###\n" + 
                "The simulation result summary of the current design: \n" + str(result_report) + 
                "\n###\n" + ## Two issues for history here: 1. will confuse model, mix history with current result (observed on deepseek) 2. No corresponding actions listed.
                "You should learn from the history of your previous tuning decisions: \n" + tuning_history_str +
                "\n###\n" + 
                "\n Based on this information, make decisions about design tuning following the output schema. Keep your thinking process short and concise!\n"
            }

            self.logger.debug(f"==== Generated prompt ==== \n {prompt_dict_to_string(prompt)} \n")

            max_retry = 3
            retry_ct = 0

            while retry_ct < max_retry:
                ## Call LLM for decision making
                response = self.llm.call_llm(prompt, self.output_schema)
                self.logger.debug(f"LLM response: {response}")
                if response is not None:
                    break
                else:
                    self.logger.warning(f"LLM response is None, retrying {retry_ct} times.")
                    retry_ct += 1

            # Get the micro design decision
            decision = response
        
            
            
            self.logger.info("Design Tuner completed successfully")
            self.logger.debug(f"Tuner decision: {decision}")
            
            return decision, result_report

            
        except Exception as e:
            error_msg = f"Unexpected error during results evaluation: {e}"
            self.logger.error(error_msg)
            raise e
        

# Usage example

if __name__ == "__main__":
    # Initialize the LLM interface
    from modules.utils.llm_interface import OpenAILLMInterface

    llm_interface = OpenAILLMInterface()
    
    
    # Initialize the demand translator
    demand_translator = DesignTuner(llm_interface)
    
    