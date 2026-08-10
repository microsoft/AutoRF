
import io
import os
import numpy as np

from pydantic import BaseModel
from contextlib import redirect_stdout


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.surface_spec_def import MetasurfaceSpec
from modules.design_info_def import DesignProposal
from modules.utils.custom_logger import setup_logger


# Resolve circuit-model resource paths relative to the repository root so the
# code works regardless of the current working directory.
_CIRCUIT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_model"
)


class CircuitSimulationCode(BaseModel):
    explanation: str
    code: str



def run_numpy_code(code_string: str) -> str:
    """
    Execute a piece of Python code that uses NumPy only,
    and return the printed output as a string.
    """
    # Create a StringIO buffer to capture the output
    buffer = io.StringIO()
    
    # Provide a restricted global namespace with just 'np'
    # so the code can use `np.array`, `np.sqrt`, etc.
    globals_dict = {
        'np': np,
        '__name__': '__main__',  # this ensures 'if __name__ == "__main__":' will be True
        }
    
    # Redirect stdout to the buffer
    with redirect_stdout(buffer):
        # Execute the code_string in the restricted global namespace
        exec(code_string, globals_dict)
    
    # Get everything that was printed
    output = buffer.getvalue()
    return output



class CircuitModelSimulator:
    def __init__(self, llm_interface):
        self.llm_interface = llm_interface

        self.logger = setup_logger()


    def get_surf_admittance(self, design_name: str):
        """
        Get the surface admittance of the metasurface for circuit model simulation.
        """
        # For now, just hard-code the surface admittance
        if design_name == "RF-Mediator":
            return (0.1+1j) * np.linspace(0.005, 0.04, 10) 
        elif "transring" in design_name.lower():
            profile_path = os.path.join(_CIRCUIT_MODEL_DIR, "profiles", "cached_S_24.0GHz.pkl")
            return f"Extract directly from the profiling files as shown in the example code. File name: `{profile_path}`"
        else:
            # Graceful fallback for templates without a pre-characterized admittance:
            # let the LLM derive it from the profiled HFSS S-parameters instead of crashing.
            self.logger.warning(
                f"No hard-coded surface admittance for template '{design_name}'. "
                "Falling back to deriving it from profiled S-parameters."
            )
            return (
                "No pre-characterized admittance range is available for this template. "
                "Derive the surface admittance from the profiled HFSS S-parameters if available, "
                "or estimate a reasonable range from the template geometry."
            )

    def get_simulator_lib_code(self, design_name: str):
        """
        Get the circuit model simulation code library.
        """
        
        # Read a python file containing the circuit model simulation code
        if design_name == "RF-Mediator":
            lib_file = os.path.join(_CIRCUIT_MODEL_DIR, "sim_lib_noPolar.py")
        elif "transring" in design_name.lower():
            lib_file = os.path.join(_CIRCUIT_MODEL_DIR, "sim_lib_polar.py")
        else:
            self.logger.warning(f"Design name {design_name} not recognized. Using default library code.")
            lib_file = os.path.join(_CIRCUIT_MODEL_DIR, "sim_lib_noPolar.py")

        with open(lib_file, "r") as f:
            file_content = f.read()

        return file_content
    
    def get_sim_results(self, code_obj: CircuitSimulationCode):
        ## run the code and get the results

        # write the code to a temp file for debugging
        with open(os.path.join(_CIRCUIT_MODEL_DIR, "temp_llm_code.py"), "w") as f:
            f.write(code_obj.code)
        
        # Execute the code
        try:
            result = run_numpy_code(code_obj.code)
        except Exception as e:
            if "line continuation" in str(e):
                self.logger.warning("Fixing the code execution error caused by unexpected character after line continuation character.")
                # likely caused by using `import numpy as np\n\ndef shunt_admittance(Y):`
                # replace double \n with single \n
                code_obj.code = code_obj.code.replace("\\n", "\n")
                result = run_numpy_code(code_obj.code)
            else:
                raise e
        self.logger.info(f"Circuit model simulation code execution result:\n {result}")

        return result


    def eval_result(self, messages, result: str):
        """
        Evaluate the simulation result and return the evaluation string, `Does it work or no`.
        """
        messages.append({
            "role": "user",
            "content": f"Here is the Simulation result by running the python code above: {result}\n Evaluate the result to answer whether the design has the potential to meet the hardware specification and what is the promising design variable values, that means any design variable leads to a functional design is good enough, for example, over 270 degrees phase range while over -6dB min amplitude. ",
        })

        # Evaluate the result
        response = self.llm_interface.get_string_response(messages)

        formatted_report = f"""
The circuit model simulator has the following results: {result}
The designer thinks {response}
"""

        return formatted_report 





    def get_guidance(self, spec: MetasurfaceSpec, design_proposal: DesignProposal):
        """
        Get guidance for the circuit model simulation.
        """

        # simulator library code for LLM to start with
        sim_lib_code = self.get_simulator_lib_code(design_proposal.template_name)

        # LLM message
        messages =[
            {
                "role": "system",
                "content": """
You are a knowledgeable wireless (RF) hardware expert specializing in metasurfaces and antennas. You will be provided with hardware specifications for a target design you need to achieve, along with description of a potential design. 
"""
            },
            {
                "role": "user",
                "content": f"""
You need to solve whether the potential design can meet the hardware specification and with what variable values it can be achieved, by writing code to simulate performance following the circuit model theory of metasurfaces.
Specifically, the circuit model theory of metasurface models each layer of metasurface as a shunt circuit component with surface admittance Y (2x2 matrix), and each substrate or air region as a transmission line, then we multiple the ABCD matrix of all components to get the simulate the S11 and S21 of the metasurface. Note that we also model the polarization of the wave in the circuit model. We can also directly calculate the ABCD matrix of the metasurface layer from the S parameters profiled from the HFSS instead of using the surface admittance Y.
You hav some library functions for circuit model simulation to start with as listed below (make sure you include them in your output).  You need to fill in the surface admittance Y of the metasurface layer or directly use the ABCD profiling results stored if available in the code, and the transmission line parameters for the air spacer and substrate. The code should simulate the S11 and S21 of the metasurface, as two 2x2 matrix to capture polarization modes if polarization is considered, and compare with the hardware specification to see if the potential design meets the requirement.
The final output of the simulation should be a string message including a list of performance values versus different variable values, and the variable values that provide good performance towards the demanded hardware specification, by printing these information at the end of your code. 
The specific printing format is decided by the target hardware specification and should follow the provided code example below.
For example, (1) `print(f"The spacer thickness with good performance is xx, xxx, with a phase range of XX degrees")` And a phase range over 270 degrees and min amplitude over -6dB is considered good performance. (2) `print(f"Best gap={{final_best_gap*1000:.1f}} mm, phase range={{np.degrees(best_phase_range):.1f}} deg")` And other code above has specified that we only consider phase range covered variations that have amplitude over -4dB.
Make sure adjust the varying range of gap between surface layers or metal sheet in the code, it should be smaller than one wavelength of the operating frequency, but at least over 25% or one quarter of wavelength to avoid coupling! for example, for 60GHz, the wavelength is 5mm, so the surface gap should be between 1.25mm and 5mm.
Note that in the code, DO NOT use double backslash for new line, just `\\n`.
###
The target metasurface specification is:
{spec}
###
The design proposal you need to test with circuit model simulator:
Reasoning comment: {design_proposal.reasoning_comment}
We want use these design strategies {design_proposal.design_strategies}. {design_proposal.strategy_comment}
###
This leads to a cascaded metasurface with the following circuit model blocks:
{design_proposal.circuit_model_blocks}
You have the following design variables to optimize:
{design_proposal.new_design_variables}
You have the following design constants to plug in your simulation:
The metasurface admittance can be tuned in the range of {self.get_surf_admittance(design_proposal.template_name)}.

###
Library code for circuit model simulation:
{sim_lib_code}

"""
            }
        ]

        # print prompt
        self.logger.debug(f"Circuit model simulation prompt: {messages[-1]}")

        response, code_obj = self.llm_interface.get_json_response(messages, CircuitSimulationCode)

        self.logger.debug(f"Circuit model simulator LLM response ==== {response}")


        messages.append({
            "role": "assistant",
            "content": response,
        })

        # Execute the code and get the results 
        result_str = self.get_sim_results(code_obj)
       
        # Evaluate the result
        report = self.eval_result(messages, result_str)

        return report


## Testing Code
if __name__ == "__main__":
    # Example usage
    from modules.utils.llm_interface import GetLLMInterface
    from modules.demand_translator import DemandTranslator

    llm_interface_o1 = GetLLMInterface(model_name="o1", model_provider="OpenRouter")
    llm_interface = GetLLMInterface(model_name="gpt-4o", model_provider="OpenAI")

    demand_translator = DemandTranslator(llm_interface)
    circuit_simulator = CircuitModelSimulator(llm_interface_o1)

    # Example user demand

    user_demand = "Design a programmable metasurface for 2.4GHz WiFi beamforming."
    # Or specify your own user demand
    spec = demand_translator.translate_demand(user_demand)

    # Example design proposal
    design_proposal = DesignProposal(
        reasoning_comment="RF-Mediator is the only metasurface in the library that operates near 2.4 GHz and supports a programmable/reconfigurable mode. However, it is transmissive and focuses on impedance rather than phase control. We can switch it to a reflective mode by adding a metal sheet behind it (OP_MODE_SWITCH). To expand its reflection phase range toward 2π coverage for beamforming, we add a duplicate metasurface layer (DUPLI_LAYERS). This creates a two-layer programmable metasurface with an air spacer and a metal backing, enabling wide-range reflection phase control suitable for 2.45 GHz Wi-Fi beamforming.",
        template_name="RF-Mediator",
        mismatched_spec=["op_mode (trans -> reflect)",
    "sig_control_type (impedance -> extended phase)",],
        design_strategies=[    "OP_MODE_SWITCH",
    "DUPLI_LAYERS"],
        strategy_comment="Add a metal sheet to turn it from transmissive to reflective and stack an extra metasurface layer to achieve a wider phase-control range needed for beamforming at 2.45 GHz.",
        circuit_model_blocks="Air - Metasurface - Air Spacer - Metal Sheet",
        new_design_variables=[
            "thickness between Metasurface and the Metal Sheet",
            "surface admittance of metasurface layer",
        ]
    )

    guidance = circuit_simulator.get_guidance(spec, design_proposal)
    print(guidance)











