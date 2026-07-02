import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path
import ansys.aedt.core
from dataclasses import dataclass
from pydantic import BaseModel
from enum import Enum

# copy files from template folder to working folder
import shutil
import os


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils.custom_logger import setup_logger


# Define constants.

AEDT_VERSION = "2024.2"
NG_MODE = False  # Open AEDT UI when it is launched.
ND_MODE = False  # Run AEDT as a new desktop window.

# Resolve the HFSS workspace folder relative to the repository root so the code
# is portable across machines. Override with the AUTORF_HFSS_WORKSPACE env var.
WORK_PLACE_FOLDER = os.environ.get(
    "AUTORF_HFSS_WORKSPACE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "HFSS_workspace"),
)


class SimulationStatus(Enum):
    SUCCESS = "success"
    # FAILED = "failed"
    HFSS_ERROR = "HFSS reported ERROR"
    CONVERGENCE_ERROR = "convergence_error"
    # MESH_ERROR = "mesh_error"
    # LICENSE_ERROR = "license_error"


class SimFigureResults(BaseModel):
    """Container for simulation figure results."""
    expression: str # The expression of the data, e.g., "dB(S11)", Y axis label
    primary_sweep: str # The primary sweep variable name, X axis label, e.g., "Freq"
    primary_sweep_values: List[float] # The primary sweep variable values, e.g., [1, 2, 3, 4, 5], X axis values
    variations: List[Dict] # The variations of the design variables, e.g., [{"R1": 0.6, "G": 0.5}, {"R1": 0.7, "G": 0.5}], Used as legends in the plot.
    data_lines: List[List] # The data samples for one or multiple lines
    fig_filename: Optional[str] = None

    def variations_str(self):
        """Return the variations as a list of strings instead of dict."""
        var_str_list = []
        for var in self.variations:
            var_str = ", ".join([f"{k}: {v*1e3}mm" for k, v in var.items()]) # Convert to mm, default unit is meter.
            var_str_list.append(var_str)
        
        return var_str_list
    
    def plot(self, title: str = "Simulation Results", save_filename : str = None):
        """Plot the simulation results."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 6))
        for i, data_line in enumerate(self.data_lines):
            plt.plot(self.primary_sweep_values, data_line, label=self.variations_str()[i])
        
        plt.xlabel(self.primary_sweep)
        plt.ylabel(self.expression)
        plt.title(title)
        plt.legend()

        # plt.show()

        fig_save_filename = self.fig_filename if save_filename is None else save_filename

        if not os.path.exists(os.path.dirname(fig_save_filename)):
            os.makedirs(os.path.dirname(fig_save_filename))
        fig.savefig(fig_save_filename)

        # plt.clf()
        plt.close(fig)
        
        

    def __str__(self):
        format_str = f"SimFigureResults: result expression `{self.expression}` with primary sweep {self.primary_sweep} and variations {self.variations_str()} \n"
        
        # Show Frequency and Y-axis values for several variations.
        # down_sample_factor = 4
        # format_str += f"Primary sweep values: {[x for x in self.primary_sweep_values[::down_sample_factor]]} \n"

        # max_var_num = min(len(self.data_lines), 1) # Only show the first 1 lines.  

        # for i, data_line in enumerate(self.data_lines[:max_var_num]):
        #     format_str += f"\nVariation#{i} Y data points: "

        #     data_line_str = ", ".join([f"{y:.2f}" for y in data_line[::down_sample_factor]])
        #     format_str += data_line_str
        
        return format_str


class SimulationResults(BaseModel):
    """Container for simulation results."""

    s_parameters: Dict[str, SimFigureResults] = None
    # bandwidth: float
    hfss_message: str = ""
    status: SimulationStatus

    def __str__(self):
        format_str = f"SimulationResults: status {self.status} \n"
        if self.s_parameters is not None:
            for expr, fig_result in self.s_parameters.items():
                format_str += str(fig_result) + "\n"
        else:
            format_str += "No S-parameter results found."
        
        return format_str

# @dataclass
class VariablePropertyInfo(BaseModel):
    """Container for variable property information."""
    name: str
    expression: str
    value: float
    unit: str
    description: str

# @dataclass
class VariableSweepInfo(BaseModel):
    """Container for variable sweep information."""

    name: str
    start: float
    stop: float
    step: float
    list_of_values: List[float] | None = None # If not None, then it is a list of values to sweep over.

class MaterialEditInfo(BaseModel):
    """Container for HFSS 3D object material edit."""
    name: str
    material: str

class ObjDeleteInfo(BaseModel):
    """Container for HFSS 3D object delete information."""
    name: str


class BoxCreateInfo(BaseModel):
    """Container for HFSS 3D Box property edit or create information."""
    name: str
    position: List[str]
    sizes: List[str]
    material: str


class MetalSheetCreateInfo(BaseModel):
    """Container for HFSS 3D Metal sheet Layer create information."""
    name: str
    position: List[str]
    sizes: List[str]

class SurfPatternCopyInfo(BaseModel):
    """Container for HFSS 3D Surface Pattern Layer create information."""
    source_obj_list: List[str] # The source pattern layer object name
    independent_pattern: bool # True if the pattern dimension is independent of the source object, False if dependent on the source object
    independent_circuit: bool # True if the circuit component variables on the pattern is independent of the source object, False if dependent
    variables_toCopy: List[str] # The variable names to copy from the source object to create new and independent variables for the pattern layer
    move_offset: List[str] # The offset of the pattern layer from the source object


@dataclass
class ObjectPropertyInfo():
    """
    Container for object property information. 
    HFSS data collection purpose, not for LLM output
    """
    name: str
    type: str
    isModel: bool
    material: str
    position: List[str]
    sizes: List[str]
    construct_history: Dict[str, Any]


def remove_unit(value_str: str) -> float:
    """Remove the unit from a string and convert to float."""
    value_idx = max([i for i, c in enumerate(value_str) if not c.isalpha()])
    return float(value_str[:value_idx + 1])


class HFSSManager:
    """Manages HFSS simulations and result processing."""

    def __init__(self):
        self.working_dir = Path(WORK_PLACE_FOLDER)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger()

        # HFSS configuration
        self.default_settings = {
            "frequency_unit": "GHz",
            "length_unit": "mm",
            "max_passes": 15,
            "max_delta_s": 0.01,
            "min_passes": 8,
        }

        self.logger.info("=== HFSS manager initialized, starting AEDT desktop === ")
        # Launch AEDT desktop
        self.desktop = ansys.aedt.core.launch_desktop(
            version=AEDT_VERSION, non_graphical=NG_MODE, new_desktop=ND_MODE
        )

    def setup_project_design(
        self,
        template_filename: str,
        new_project_filename: str = "new_surface_default.aedt",
        design_solution_type: str = "Modal",
    ) -> None:
        """Set up active project and design in HFSS. may be new or copied from existing."""

        self.logger.info(f"Setting up HFSS project and design {new_project_filename}")

        # Open the project and design
        project_name = str(self.working_dir / new_project_filename)
        if not os.path.exists(project_name):
            self.logger.info(f"Creating new project {new_project_filename} from template {template_filename}.")
            # Copy template design from the subfolder
            template_folder = self.working_dir / "design_templates"
            # Move the target design from the template folder to the working folder
            try:
                shutil.copy(
                    template_folder / template_filename,
                    self.working_dir / new_project_filename,
                )
            except Exception as e:
                self.logger.error(f"Failed to copy template design: {str(e)}")
                raise e


        # Provides the HFSS application interface.
        # This class allows you to create an interactive instance of HFSS and connect to an existing HFSS design
        # or create a new HFSS design if one does not exist.
        self.hfss = ansys.aedt.core.Hfss(
            version=AEDT_VERSION,
            project=project_name,
            design=None,  # find a default design in the project
            # solution_type=self.default_settings["solver_type"],
            solution_type=design_solution_type,
        )

    # For teating purpose ONLY, not supposed to call all the steps together.
    def setup_simulation(self) -> None:
        """Set up HFSS simulation environment with design parameters."""
        try:
            self.logger.info("Setting up HFSS simulation")

            # Set up materials and geometry
            # self._create_geometry(hfss, design_params)

            # Set up boundaries
            # self._setup_boundaries(hfss, design_params)

            # Insert 3D component
            # self._insert_3d_component()

            # Set up analysis setup
            # self._setup_analysis()

            # Set up param sweeps
            # self._setup_param_sweeps()

            # Set up reports
            # self._setup_reports()

        except Exception as e:
            self.logger.error(f"Failed to set up HFSS simulation: {str(e)}")
            # raise SimulationSetupError(str(e))
            raise e

    def _insert_3d_component(self, unitcell_path) -> None:

        hfss = self.hfss
        # params = self.design_params

        try:
            # unitcell_path = params.get('3d_component', "None")
            comp = hfss.modeler.insert_3d_component(unitcell_path)

            # Assign the parameter to the 3D component.

            # component_name = hfss.modeler.user_defined_component_names
            # comp.parameters["a"] = "patch_dim"

        except Exception as e:
            # raise SimulationSetupError(f"Failed to insert 3D component: {str(e)}")
            raise e


    # def _setup_boundaries(self, hfss: ansys.aedt.core.Hfss, params: Dict[str, Any]) -> None:
    #     """Set up simulation boundaries."""
    #     try:
    #         # Get region dimensions
    #         region = hfss.modeler.get_object_from_name("Region")
    #         bbox = region.bounding_box

    #         # Assign lattice pairs for periodic structure
    #         hfss.auto_assign_lattice_pairs(assignment=region.name)

    #         # Create Floquet ports
    #         hfss.create_floquet_port(
    #             assignment=region.top_face_z,
    #             lattice_origin=[0, 0, bbox[5]],  # z_max
    #             lattice_a_end=[0, bbox[4], bbox[5]],  # [0, y_max, z_max]
    #             lattice_b_end=[bbox[3], 0, bbox[5]],  # [x_max, 0, z_max]
    #             name="FloquetPort_Top",
    #             deembed_distance=params.get('port_deembed', 0)
    #         )

    #     except Exception as e:
    #         raise SimulationSetupError(f"Failed to set up boundaries: {str(e)}")

    def setup_analysis(self, analysis_params: Dict[str, Any]) -> None:
        """Configure analysis settings."""
        hfss = self.hfss
        params = analysis_params

        self.logger.info("Setting up HFSS analysis")

        try:
            # Get Current setup
            # setup = hfss.create_setup("MySetup")
            setup_names = hfss.get_setups()
            assert len(setup_names) > 0, "No setup found in the project"
            setup = hfss.get_setup(setup_names[0])

            # Update setup settings
            # setup.props["SolveType"] = "Broadband"
            # setup.enable_adaptive_setup_broadband(low_frequency="10GHz", high_frquency="40GHz") # original API has typo...
            # NOTE: Lacking support for broadband setup in HFSS 2024.2 (At least I failed to get it working...)
            setup.props["Frequency"] = f"{params['frequency_range'][1]}GHz"
            setup.props["MaxDeltaS"] = params.get(
                "max_delta_s", self.default_settings["max_delta_s"]
            )
            setup.props["MaximumPasses"] = params.get(
                "max_passes", self.default_settings["max_passes"]
            )
            setup.props["MinimumPasses"] = params.get(
                "min_passes", self.default_settings["min_passes"]
            )

            setup.update()
            self.logger.debug(f"Updated setup Frequency = {setup.props["Frequency"]}")

            # Update the all frequency sweeps in the setup
            # For antenna templates, they have multiple frequency sweeps.
            for id in range(len(setup.sweeps)):
                sweep = setup.sweeps[id]
                sweep.props["Type"] = "Interpolating"
                extra_freq_value = (
                    0  # 1GHz, extra frequency value to ensure the range is covered
                )
                lower_freq = params["frequency_range"][0] - extra_freq_value
                sweep.props["RangeStart"] = (
                    str(lower_freq) + "GHz" if lower_freq > 0 else "0GHz"
                )
                sweep.props["RangeEnd"] = (
                    str(params["frequency_range"][1] + extra_freq_value) + "GHz"
                )
                if params["frequency_range"][1] < 10:
                    per_ghz_points = 100  # Points per GHz, resolution is 10 MHz
                else:
                    per_ghz_points = 20 # Points per GHz, resolution is 50 MHz
                sweep.props["RangeCount"] = (
                    2 * extra_freq_value
                    + params["frequency_range"][1]
                    - params["frequency_range"][0]
                ) * per_ghz_points + 1 # Add one to include the end point

                sweep.update()
                self.logger.debug(f"Updated setup Sweep#{id} frequency range = [{sweep.props["RangeStart"]}, {sweep.props["RangeEnd"]}] ")

            # Create frequency sweep
            # hfss.create_linear_count_sweep(
            #     setup=setup.name,
            #     units="GHz",
            #     start_frequency=6,
            #     stop_frequency=15,
            #     num_of_freq_points=51,
            #     name="sweep1",
            #     sweep_type="Interpolating",
            #     interpolation_tol=6,
            #     save_fields=False,
            # )

        except Exception as e:
            # raise SimulationSetupError(f"Failed to set up analysis: {str(e)}")
            raise e

    def read_design_variables(self) -> List[Dict]:
        """Read and return design variables with their descriptions."""
        hfss = self.hfss

        # variables = hfss.design_variables # This will return a dictionary of design object variables
        all_variables = hfss.variable_manager.variables
        variable_list = []
        for var_name, var_details in all_variables.items():
            variable_list.append(
                # {
                #     "Name": var_name,
                #     "Value": var_details.numeric_value,
                #     "Unit": var_details.units,
                #     "Description": var_details.description,
                # }
                VariablePropertyInfo(
                    name=var_name,
                    expression=var_details.expression,
                    value=var_details.numeric_value,
                    unit=var_details.units,
                    description=var_details.description,
                )
            )
        return variable_list

    def update_design_variables(self, params_to_update: List[VariablePropertyInfo]) -> None:
        """
        Update design variables with new values.
        Args:
            self:
            params_to_update: Dictionary of parameter names and their new values.
        """
        hfss = self.hfss
    
        for var_info in params_to_update:
            if var_info.name not in hfss.variable_manager.variables:
                self.logger.debug(f"Variable {var_info.name} not found in HFSS design variables")
                # switch between lower and upper cases, and try again
                var_info.name = var_info.name.lower() if var_info.name.isupper() else var_info.name.upper()
                # If still not found, skip this variable.
                if var_info.name not in hfss.variable_manager.variables:
                    self.logger.warning(f"Variable {var_info.name} not found in HFSS design variables even after converting lower and upper cases. So skip this variable.")
                    continue                

            # hfss[var_info.name] = str(var_info.value) + var_info.unit  # This will overwrite the variable description, which is not desired.
            hfss.variable_manager.variables[var_info.name].expression = str(var_info.value) + var_info.unit
                # (str(var_info.value) + var_info.unit) if var_info.expression == '' else var_info.expression (sometimes expression is '0.4 * 0.4 * 0.925')

            self.logger.debug(f"Updated design variable {var_info.name} to {hfss[var_info.name]}")
            if hfss[var_info.name] != str(var_info.value) + var_info.unit:
                self.logger.warning(f"Updated design variable {var_info.name} to {hfss[var_info.name]} is different from the desired value {str(var_info.value) + var_info.unit}")
                # try again?
                hfss.variable_manager.variables[var_info.name].expression = str(var_info.value) + var_info.unit

    def create_design_variables(self, params_to_create: List[VariablePropertyInfo]) -> None:
        """
        Create new design variables with new values or expression.
        Args:
            self:
            params_to_update: Dictionary of parameter names and their new values.
        """
        hfss = self.hfss
    
        for var_info in params_to_create:
            assert var_info.name not in hfss.variable_manager.variables, f"Variable {var_info.name} already found in HFSS design variables"

            hfss[var_info.name] = var_info.expression if var_info.expression != '' else str(var_info.value) + var_info.unit

            self.logger.debug(f"Created design variable {var_info.name} as {hfss[var_info.name]}")

    def read_parameter_sweep(self, sweep_id: int = 0) -> Tuple[List, Dict]:
        """Read and return details of a parameter sweep. By default, reads the first sweep."""
        hfss = self.hfss

        # Get the sweep by the ID
        if len(hfss.parametrics.setups) > 0:
            sweep = hfss.parametrics.setups[sweep_id]
            assert sweep is not None, "Failed to read parameter sweep"
            assert sweep.soltype == "OptiParametric", "Unexpected sweep type"
        else:
            self.logger.debug("No param sweep info available, return None.")
            return None, None

        # Get the sweep details
        sweep_details = {
            "name": sweep.name,
            "isEnable": sweep.props["IsEnabled"],
            "variable_sweeps": sweep.props["Sweeps"]["SweepDefinition"], # main info about what variable sweeps.
        }

        # Parse the variable sweeps definitions as VariableSweepInfo format
        var_sweep_info = []
        # sweep_details["variable_sweeps"] can be a list of dict or a single dict, we convert it to list of dict for consistency, so we check the type of it first.
        hfss_sweep_list = sweep_details["variable_sweeps"] if isinstance(sweep_details["variable_sweeps"], list) else [sweep_details["variable_sweeps"]]

        # Loop over all variables in the sweep.
        for var_sweep in hfss_sweep_list:
            # parse var_sweep['Data'] with the format of `LIN 0.6mm 1.1mm 0.1mm` to get the numbers as start, stop, and step values
            # assert (
            #     var_sweep["Data"].split(" ")[0] == "LIN"
            # ), "Only linear sweep supported, but got " + var_sweep["Data"]

            # Parse the sweep info to get variation values
            if var_sweep["Data"].split(" ")[0] == "LIN":
                # linear sweep
                start, stop, step = var_sweep["Data"].split(" ")[1:]
                # find the max index that is not a-z to remove the unit, e.g., "0.6mm" -> "0.6"
                start, stop, step = remove_unit(start), remove_unit(stop), remove_unit(step)
                list_of_values = None
            else:
                # tunable capacitor are non-uniform single values only.
                list_of_values =  [remove_unit(x) for x in var_sweep["Data"].split(" ")]
                start, stop, step = list_of_values[0], list_of_values[-1], "0" # dummy value
            

            var_sweep_info.append(
                VariableSweepInfo(
                    name=var_sweep["Variable"], start=start, stop=stop, step=step, list_of_values=list_of_values
                )
            )

        return var_sweep_info, sweep_details

    def delete_parameter_sweep(self, sweep_id: int = 0) -> None:
        """
        Delete design variables with new values.
        """
        hfss = self.hfss

        if len(hfss.parametrics.setups) == 0:
            return # No sweep to delete

        sweep = hfss.parametrics.setups[sweep_id]
        sweep.delete()
        self.logger.debug(f"Deleted sweep#{sweep_id}")

    def create_parameter_sweep(
        self, sweep_name: str, param_ranges: list[VariableSweepInfo]
    ):
        """
        Create a parameter sweep.
        Args:
            self:
            sweep_name: Name of the parametric sweep.
            param_ranges: Dictionary of parameters and their ranges in the form:
                        {"ParamName": {"Start": value, "Stop": value, "Step": value}}
        """
        hfss = self.hfss

        assert len(param_ranges) > 0, "No parameter ranges provided"

        # Create a parametric analysis
        hfss.parametrics.add(
            name=sweep_name,
            variable=param_ranges[0].name,
            start_point=param_ranges[0].start,
            end_point=param_ranges[0].stop,
            step=param_ranges[0].step,
            variation_type="LinearStep",
        )

        assert (
            hfss.parametrics.setups[-1].name == sweep_name
        ), "Failed to create parametric sweep"

        for i in range(1, len(param_ranges)):
            hfss.parametrics.setups[-1].add_variation(
                sweep_variable=param_ranges[i].name,
                start_point=param_ranges[i].start,
                end_point=param_ranges[i].stop,
                step=param_ranges[i].step,
                variation_type="LinearStep",
            )

        self.logger.debug(f"Added sweep {sweep_name} : \n {param_ranges}")

    # Testing purpose only
    def setup_param_sweeps(self):
        """
        Testing Set up parametric sweeps.

        Should use the functions separately:

        (1) read_design_variables, (2) update_design_variables, (3) read_parameter_sweep, (4) delete_parameter_sweep, (5) create_parameter_sweep
        """
        # (1) Read design variables
        variables = self.read_design_variables()
        print("Design variables:", variables)

        # (2) Update nominal parameters
        params_to_update = [
            VariablePropertyInfo(name="R1", value=0.6, unit="mm", description=""),
            VariablePropertyInfo(name="G", value=0.3, unit="mm", description=""),
        ]
        self.update_design_variables(params_to_update)

        # (3) Read old parameter sweep
        var_info_list, sweep_details = self.read_parameter_sweep()
        print(var_info_list)
        # OPTIONAL: Delete old parameter sweep
        self.delete_parameter_sweep(-1)

        # (4) Create a parameter sweep
        param_ranges = [
            VariableSweepInfo(
                name="R1", start=0.6, stop=1.1, step=0.1
            ),  # Will use the variable's default unit
            VariableSweepInfo(name="G", start=0.3, stop=0.5, step=0.2),
        ]
        self.create_parameter_sweep("Sweep1", param_ranges)

    def _get_Sparam_expr(self, S_expr_list = ["S11"], polar_mode=["2", "2"]):
        # Create separate S-parameter plots for each port
        ports = self.hfss.ports  # Get all port names
        

        # Create S-parameter expressions
        if self.hfss.solution_type == "Modal":
            # S_expr_list = ["S11", "S21"]
            expr_list = []
            for expr_str in S_expr_list:
                if expr_str == "S11":
                    expr_list.append(f"S({ports[0]}:{polar_mode[0]},{ports[0]}:{polar_mode[1]})")
                elif expr_str == "S21":
                    expr_list.append(f"S({ports[1]}:{polar_mode[0]},{ports[0]}:{polar_mode[1]})")
                else:
                    raise ValueError(f"Invalid S-parameter expression: {expr_str}")
        elif self.hfss.solution_type == "Terminal":
            expr_list = []
            for expr_str in S_expr_list:
                if expr_str == "S11":
                    expr_list.append(f"St11")
                else:
                    raise ValueError(f"Invalid Terminal S-parameter expression: {expr_str}")

        # variations = {"Freq": ["All"]}
        variations = {}
        
        # Read the variable names from the parameter sweep as variations
        var_info_list, _ = self.read_parameter_sweep()
        if var_info_list is not None:
            for var_info in var_info_list:
                variations[var_info.name] = ["All"]
        else:
            variations = {"Freq": ["All"]} # or None?

        return expr_list, variations

    def setup_reports(self) -> None:
        # ## Postprocess
        #
        # Create S-parameter reports.

        hfss = self.hfss

        S_expr_list, expr_list, variations = self._get_Sparam_expr(S_expr_list, polar_mode=self._get_polar_mode())
        
        for e_id in range(len(expr_list)):
            plot_name = f"{S_expr_list[e_id]} Mag"
            hfss.post.create_report(
                expressions=f"dB({expr_list[e_id]})",  # Replace with your specific S-matrix elements
                primary_sweep_variable="Freq",
                variations=variations,
                report_category="Modal Solution Data",
                plot_name=plot_name,
            )
            plot_name = f"{S_expr_list[e_id]} Phase"
            hfss.post.create_report(
                expressions=f"ang_deg({expr_list[e_id]})",  # Replace with your specific S-matrix elements
                primary_sweep_variable="Freq",
                variations=variations,
                report_category="Modal Solution Data",
                plot_name=plot_name,
            )
            self.logger.debug(f"Generated plot: {plot_name}")

        # + Plot all Quantities
        # all_quantities = hfss.post.available_report_quantities()

    def run_simulation(self, S_expr_list: List[str] = ["S11"]) -> SimulationResults:
        """Execute HFSS simulation and gather results."""

        self.logger.info("Starting HFSS simulation")

        hfss = self.hfss

        # Validate project and design
        validation_message, validation_passed = hfss.validate_full_design()
        validation_message = "\n".join(validation_message)

        if not validation_passed:
            self.logger.warning(f"HFSS Design validation failed: {validation_message}")
            # self.cleanup()
            return SimulationResults(status=SimulationStatus.HFSS_ERROR, hfss_message=validation_message)
        else:
            self.logger.info("HFSS Design validation passed, proceeding with simulation...")

        # Wrong code below, should be removed after the validation is fixed.
        # assert hfss.validate_full_design(), "Design validation failed"

        # Run analysis
        ret = hfss.analyze(
            setup = None, # solve all setups
            cores = 20, # Use 12 cores (depends on the machine. Do not use up all cores to avoid OS freeze)
            tasks = 5, # Use 4 tasks distributed (3 cores per task)
            # use_auto_settings = False,
        )
        if ret == False:
            self.logger.error("HFSS simulation failed! Checking the HFSS error messages...")
            # Get HFSS error message
            msg_list = self.read_aedt_messages(level="error")
            design_msg = "".join(msg_list)
            return SimulationResults(status=SimulationStatus.HFSS_ERROR, hfss_message=f"HFSS simulation failed with error. Messages: {design_msg}")

        # Save project after analysis
        hfss.save_project()

        # Gather results
        results = self._gather_results(S_expr_list=S_expr_list)

        return results

    def _get_fig_result_by_expr(self, cur_expr: str, primary_var: str, variations: list) -> SimFigureResults:
        """Get results of all variations, formatted as the SimFigureResults class, based on the input expression."""
        self.logger.debug(f"Getting HFSS results with expression: {cur_expr}")

        result_obj = self.hfss.post.get_solution_data(
            expressions=cur_expr,
            primary_sweep_variable=primary_var,
            variations=variations,
            report_category="Modal Solution Data",
        )
        # Plot does not work for unknown reason.
        # result_obj.plot(title=quantity)

        fig_results = SimFigureResults(expression=cur_expr, 
                                       primary_sweep=primary_var, 
                                       primary_sweep_values=result_obj.primary_sweep_values,
                                       variations=result_obj.variations, 
                                       data_lines=[],
                                       fig_filename= WORK_PLACE_FOLDER +R"\\sim_figs\\" + self.hfss.project_name + f"_{cur_expr.replace(":", "_")}.png"
                                       )
        for id, var in enumerate(result_obj.variations):
            self.logger.debug(f"Variation {id}: {var}")
            
            result_obj.set_active_variation(id)
            assert result_obj.active_variation == var, f"Failed to set active variation to {id}"
            fig_results.data_lines.append(result_obj.data_real())
        
        fig_results.plot()
        self.logger.debug(f"Saved Sim result figure of {fig_results.expression} to {fig_results.fig_filename}")
        
        return fig_results
    
    def _get_polar_mode(self):
        """
        Different projects may use different polar modes.
        For now, we hard code the polar mode based on the project name. Not a good practice.
        """

        # Return the polar mode based the project name.
        design_name = self.hfss.design_name.lower()
        self.logger.info(f"Design name: {design_name} ")
        if "AutoMS".lower() in design_name or "RFocus".lower() in design_name:
            polar_mode = ["2", "2"]
        elif "RF-mediator".lower() in design_name:
            polar_mode = ["1", "1"]
        elif "TransRing".lower() in design_name:
            polar_mode = ["1", "2"]
        else:
            assert False, f"Unknown design name: {design_name}. Please check the design name and update the polar mode."

        self.logger.info(f"Using polar mode: {polar_mode}")
        
        return polar_mode


    def _gather_results(self, S_expr_list=["S11"]) -> SimulationResults:
        """Extract and process simulation results."""
        # Get available quantities
        # quantities = hfss.post.available_report_quantities()

        # S_expr_list = ["S11"]
        expr_list, variations = self._get_Sparam_expr(S_expr_list, polar_mode=self._get_polar_mode())
        # Can also check variations with `self.hfss.available_variations``, but it is not useful as it just gives all variables.

        # Get S-parameters
        results = {}
        primary_var = "Freq"
        for id, quantity in enumerate(expr_list):
            ## 1. Get magnitude as dB values
            cur_expr = f"dB({quantity})"
            results[S_expr_list[id] +" Mag"] = self._get_fig_result_by_expr(cur_expr, primary_var, variations)


            # cur_expr = f"cang_deg({quantity})" # Continuous phase angle values, no wrap.
            cur_expr = f"ang_deg({quantity})" # `cang` will cause strange error, can not get all variations.
            results[S_expr_list[id]+ " Phase"] = self._get_fig_result_by_expr(cur_expr, primary_var, variations)


        return SimulationResults(s_parameters=results, status=SimulationStatus.SUCCESS)
    

    def _get_3D_object_properties(self, obj_name) -> ObjectPropertyInfo:
        """Read and return object properties of the input 3D object."""
        hfss = self.hfss

        # Get the object with the name
        obj = hfss.modeler.get_object_from_name(obj_name)

        # Read the history object that stores all the operation history when creating the object.
        # obj_history = obj.history()
        obj_history_dict = obj.history().jsonalize_tree()[obj_name] # A dictionary that stores the tree-structure history of the object.

        # make sure the command starts with "CreateXX"
        assert obj_history_dict['Props']['Command'].startswith("Create"), "The object is NOT created by a command that starts with 'Create'."

        if "Position" in obj_history_dict['Props']:
            obj_position = [str(x) for x in obj_history_dict['Props']['Position'] ] # Must be str, convert if not str.
        elif "Center Position" in obj_history_dict['Props']:
            # if the object is a circle, the position is the center of the circle.
            obj_position = [str(x) for x in obj_history_dict['Props']['Center Position'] ] 
        elif obj_history_dict['Props']['Command'] == "CreateRegion":
            # Region object does not have position, set to empty string.
            obj_position = ["", "", ""]
        else:
            assert False, "The object does NOT have a position or center position."

        obj_sizes = [
            obj_history_dict['Props'].get('XSize', ''), # If XSize is not in the history, it means the object is 2D. So we set it to empty string.
            obj_history_dict['Props'].get('YSize', ''), # If YSize is not in the history, it means the object is 2D. So we set it to empty string.
            obj_history_dict['Props'].get('ZSize', '') # If ZSize is not in the history, it means the object is 2D. So we set it to empty string.
            ] 
        
        obj_info = ObjectPropertyInfo(
            name=obj.name,
            type=obj.object_type,
            isModel=obj.model,
            material=obj.material_name,
            position=obj_position,
            sizes=obj_sizes,
            construct_history=obj_history_dict,
        )

        self.logger.debug(f"Object {obj_name} info: {obj_info}")

        return obj_info

    def _oeditor_set_property(self, obj_name: str, command_name: str, prop_name: str, prop_value: str | List[str]) -> bool:
        """
        Set the property of an object in HFSS 3D modeler.
        """
        if prop_name != "Position" and prop_name != "Center Position" and prop_name != "Vector":
            self.logger.warning(f"Unknown property name: {prop_name}")

        if "/" in prop_name:
            self.logger.warning(f"Property name {prop_name} contains '/', which is not a legit prop.")
            return False

        # If the object name contains a colon, remove the part after the colon.
        if ":" in obj_name:
            obj_name = obj_name.split(":")[0]
        
        self.logger.debug(f"Setting property {command_name}:{prop_name} of {obj_name} to {prop_name} = {prop_value}")
        try:
            if isinstance(prop_value, list):
                assert len(prop_value) == 3, "The property value must be a list of 3 elements for X, Y, Z."
                ret = self.hfss.modeler.oeditor.ChangeProperty(
                    [
                        "NAME:AllTabs",
                        [
                            "NAME:Geometry3DCmdTab",
                            [
                                "NAME:PropServers", 
                                f"{obj_name}:{command_name}:1"
                            ],
                            [
                                "NAME:ChangedProps",
                                [
                                    f"NAME:{prop_name}",
                                    "X:="			, prop_value[0],
                                    "Y:="			, prop_value[1],
                                    "Z:="			, prop_value[2]
                                ]
                            ]
                        ]
                    ])
            else:
                ret = self.hfss.modeler.oeditor.ChangeProperty(
                    [
                        "NAME:AllTabs",
                        [
                            "NAME:Geometry3DCmdTab",
                            [
                                "NAME:PropServers", 
                                f"{obj_name}:{command_name}:1"
                            ],
                            [
                                "NAME:ChangedProps",
                                [
                                    f"NAME:{prop_name}",
                                    "Value:=", prop_value
                                ]
                            ]
                        ]
                    ])

            if ret == False:
                self.logger.error(f"Failed to set property {prop_name}:{command_name} of {obj_name} to {prop_value}")
            return ret

        except Exception as e:
            self.logger.error(f"Failed to set property {prop_name}:{command_name} of {obj_name} to {prop_value}: Due to Error -> {str(e)}")
            return False


    def _update_3D_object_variable(self, obj_name: str, var_name_old: str, var_name_new: str) -> bool:
        """
        Go over the object history and replace the provided variable name to a new one.
        """
        hfss = self.hfss

        # Get the object with the name
        obj = hfss.modeler.get_object_from_name(obj_name)

        # Read the history object that stores all the operation history when creating the object.
        # obj_history = obj.history()
        obj_history_dict = obj.history().jsonalize_tree()[obj_name] # A dictionary that stores the tree-structure history of the object.

        # if var_name_old not in obj_history_dict['Props']['Command']:
        # return Early

        # Check the props of all objects in the history using BFS, and replace the variable name if find a match
        # --- Each prop corresponds to one editing operation in HFSS
        obj_list = [obj_history_dict] # a queue that stores the object properties
        obj_name_list = [obj_name] # a queue that stores the object names
        # BFS, search all objects in the history, until all objects are checked.
        while len(obj_list) > 0:
            cur_obj_name = obj_name_list.pop()
            cur_obj = obj_list.pop()
            cur_props = cur_obj['Props']
            # self.logger.debug(f"Check object {cur_obj_name}:{cur_props['Command']}")            # Check the properties of the current object
            for key, value in cur_props.items():
                # If the value is a list, for `Position` and `Center Position`, we need to check each element in the list.
                if isinstance(value, list):
                    new_value_list = []
                    replace_flag = False
                    for item in value:
                        if isinstance(item, str) and var_name_old == item:
                            new_value_list.append(item.replace(var_name_old, var_name_new))
                            replace_flag = True
                        else:
                            new_value_list.append(item)
                    if replace_flag:
                        assert len(new_value_list) == 3, f"Failed to replace the list {value} with {new_value_list}"
                        self._oeditor_set_property(cur_obj_name, cur_props['Command'], key, new_value_list)
                elif isinstance(value, str) and var_name_old == value:
                    # Replace the variable name
                    new_value = value.replace(var_name_old, var_name_new)
                    self._oeditor_set_property(cur_obj_name, cur_props['Command'], key, new_value)
            
            # Add the children objects to the queue
            if 'Children' in cur_obj:
                for child_name, child_obj in cur_obj['Children'].items():
                    # self.logger.debug(f"Add child {child_name}:{child_obj["Props"]["Command"]} to the queue")
                    obj_name_list.append(child_name)
                    obj_list.append(child_obj)



    def read_design_objects_info(self) -> List[ObjectPropertyInfo]:
        """Read and return all the 3D design objects information of the HFSS design. """
        
        # Get all object names in the design 3D modeler
        object_names = self.hfss.modeler.object_names
        object_info_list = []

        for obj_name in object_names:
            object_info_list.append(self._get_3D_object_properties(obj_name))

        return object_info_list


    def set_design_object_material(self, obj_name: str, material_name: str) -> bool:
        """ Set the material of a design 3D object. """

        try:
            ret = self.hfss.modeler.oeditor.SetPropertyValue("Geometry3DAttributeTab", obj_name, "Material", f"\"{material_name}\"") # there must be quote marks in the string... 
        
        except Exception as e:
            self.logger.error(f"Failed to set material of {obj_name} to {material_name}: Due to Error -> {str(e)}")
            return False

        # Magic hfss syntax to set the material of an object. Using the shorter version above.
        # ret = self.hfss.modeler.oeditor.ChangeProperty(
        #     [
        #         "NAME:AllTabs",
        #         [
        #             "NAME:Geometry3DAttributeTab",
        #             [
        #                 "NAME:PropServers", obj_name
        #             ],
        #             [
        #                 "NAME:ChangedProps",
        #                 ["NAME:Material", "Value:=", f"\"{material_name}\""]  # New material
        #             ]
        #         ]
        #     ]
        # )

        if ret == False:
            self.logger.error(f"Failed to set material of {obj_name} to {material_name}")
        
        return ret
    
    def delete_design_object(self, obj_name: str) -> bool:
        """ Delete a design 3D object from the HFSS design. """

        # longer version of deleting an object
        # ret = self.hfss.modeler.oeditor.Delete(["NAME:Selections", "Selections:=", obj_name])

        assert obj_name is not None, "Object name is None"

        ret = self.hfss.modeler.delete(obj_name) # This is easier to use than oeditor. 
        if ret == False:
            self.logger.error(f"Failed to delete {obj_name}")
        
        return ret
    
    def create_box_object(self, obj_name: str, position: List[str], sizes: List[str], material_name: str) -> bool:
        
        # Set to Global coordinate system
        self.hfss.modeler.set_working_coordinate_system("Global") # Set to Global coordinate system

        # Create a box object with the given name, position, sizes, and material.
        ret = self.hfss.modeler.create_box(
            origin=position,
            sizes=sizes,
            name=obj_name,
            material=material_name,
        )

        if not ret:
            self.logger.error(f"Failed to create box object {obj_name}")
        return ret
    

    def create_metal_sheet(self, create_info: MetalSheetCreateInfo) -> bool:

        obj_name, position, sizes = create_info.name, create_info.position, create_info.sizes

        self.logger.debug(f"Creating Metal sheet object {obj_name} at position {position} with sizes {sizes}")
        
        # Set to Global coordinate system
        self.hfss.modeler.set_working_coordinate_system("Global") # Set to Global coordinate system

        # Create a box object with the given name, position, sizes, and material.
        ret = self.hfss.modeler.create_box(
            origin=position,
            sizes=sizes,
            name=obj_name,
            material="copper",
        )

        if not ret:
            self.logger.error(f"Failed to create Metal sheet object {obj_name}")
        return ret
    
    def create_dup_pattern(self, create_info: SurfPatternCopyInfo) -> bool:
        """
        Duplicated surface pattern can not have independent pattern variables, only circuit variables are independent. so not very useful for us.
        """
        
        # Set to Global coordinate system
        self.hfss.modeler.set_working_coordinate_system("Global") # Set to Global coordinate system

        ret = self.hfss.modeler.duplicate_along_line(
            assignment=create_info.source_obj_list,
            vector=create_info.move_offset,
            attach= False if create_info.independent_circuit else True,
            clones=2, # 2 copies including the original one.
        )
        if not ret:
            self.logger.error(f"Failed to duplicate pattern [{create_info.source_obj_list}] along line with offset {create_info.move_offset}")
        return ret

    
    def copy_surf_objects(self, copy_info: SurfPatternCopyInfo) -> bool:
        """
        Make a totally independent copy of the surface objects by Copy and Paste.
        """

        self.logger.debug(f"Copying surface objects [{copy_info.source_obj_list}] to a new position {copy_info.move_offset}")

        # Set to Global coordinate system
        self.hfss.modeler.set_working_coordinate_system("Global")

        # 1. Copy the source objects
        ret = self.hfss.modeler.copy(copy_info.source_obj_list)
        if not ret:
            self.logger.error(f"Failed to copy objects [{copy_info.source_obj_list}]")
            return False
    
        # 2. Paste the copied objects
        ret = self.hfss.modeler.paste()
        if not ret:
            self.logger.error(f"Failed to paste copied objects")
            return False
        else:
            self.logger.debug(f"Successfully copied and pasted objects [{copy_info.source_obj_list}]")


        # 3. create new variables
        var_name_pairs = []
        if copy_info.independent_pattern and copy_info.variables_toCopy is not None:
            for var_name in copy_info.variables_toCopy:
                # (1) verify that the variable exists
                if var_name not in self.hfss.variable_manager.variables:
                    self.logger.warning(f"Variable {var_name} not found in HFSS design variables")
                    continue
                # (2) copy the variable as a new variable with a suffix "_1" or "_2" if "_1" already exists.
                new_var_name = var_name + "_1"
                if new_var_name in self.hfss.variable_manager.variables:
                    new_var_name = var_name + "_2"
                assert new_var_name not in self.hfss.variable_manager.variables, f"Variable {new_var_name} already found in HFSS design variables"
                self.hfss[new_var_name]= self.hfss.variable_manager.variables[var_name].expression

                self.logger.debug(f"Created new design variable {new_var_name} as {self.hfss.variable_manager.variables[new_var_name].expression}")
                var_name_pairs.append([var_name, new_var_name])
        
        # 4. Move the pasted objects to the new position
        # Get the list of pasted object names, typically "XX_1", added a number suffix to the original object names.
        pasted_obj_list = [obj_name + "1" for obj_name in copy_info.source_obj_list]
        # test if the pasted objects are created successfully.
        for obj_name in pasted_obj_list:
            if obj_name not in self.hfss.modeler.object_names:
                self.logger.warning(f"Failed to find object {obj_name} for move operation.")
                continue
        
            # 4.1. Move the pasted objects to the new position
            ret = self.hfss.modeler.move(
                assignment=obj_name,
                vector=copy_info.move_offset)
            if not ret:
                self.logger.error(f"Failed to move pasted objects [{pasted_obj_list}] to the new position {copy_info.move_offset}")


        # 5. Update the 3D surface pattern variables to new names
        for obj_name in pasted_obj_list:
            for var_name_old, var_name_new in var_name_pairs:
                self._update_3D_object_variable(obj_name=obj_name, var_name_old=var_name_old, var_name_new=var_name_new)

        # 6. Add new variables to the parameter sweep
        self.logger.debug(f"Adding new variables to the parameter sweep: {var_name_pairs}")
        # (1) read existing parameter sweep 
        var_sweep_info, sweep_details = self.read_parameter_sweep()

        # (2) check all new variables to see whether they are in the sweep or not.
        for var_pair in var_name_pairs:
            var_name_old, var_name_new = var_pair
            # If in the sweep, add the new variable to the sweep.
            for var_sweep in var_sweep_info:
                if var_sweep.name == var_name_old:
                    # Add the new variable to the sweep
                    var_sweep_info.append(
                        VariableSweepInfo(
                            name=var_name_new, start=var_sweep.start, stop=var_sweep.stop, step=var_sweep.step, list_of_values=var_sweep.list_of_values
                        )
                    )
                    self.logger.debug(f"Added new variable {var_name_new} to the parameter sweep")
                    break
        
        # (3) create a new parameter sweep with the new variables
        self.delete_parameter_sweep(sweep_id=-1)
        self.create_parameter_sweep("Sweep1", var_sweep_info)
        
        return ret
    

    def _update_design_object_geometry(self) -> None:
        """Instead of updating the geometry, we should use other function to delete and create a new one."""
        # Some examples below
        self.hfss.modeler.oeditor.GetProperties("Geometry3DCmdTab", "Box3:CreateBox:1")

        self.hfss.modeler.oeditor.GetPropertyValue("Geometry3DCmdTab", "Box3:CreateBox", "ZSize")

        self.hfss.modeler.oeditor.SetPropertyValue("Geometry3DCmdTab", "Box2:CreateBox:1", "ZSize", "10mm")

        assert False, "Should not be here."

    def read_aedt_messages(self, level : str ="error") -> List[str]:
        """Read and return all the messages in the HFSS AEDT message window."""
        try:
            msg_obj = self.hfss.logger.aedt_messages
            if level == "error":
                # All error messages, global and design
                return msg_obj.error_level
            elif level == "warning":
                # All warning messages, global and design
                return msg_obj.warning_level
            elif level == "design":
                # All messages for the current active design
                return msg_obj.design_level
            elif level == "global":
                # All global messages, not very useful for now.
                return msg_obj.global_level
        except Exception as e:
            self.logger.error(f"Failed to read AEDT messages: {str(e)}")
            # try using another way of getting messages
            return self.hfss.logger.aedt_error_messages
        
    def clear_aedt_messages(self) -> None:
        """Clear all the messages in the HFSS AEDT message window."""
        try:
            # Clear all messages, all designs, all level, including global messages.
            self.hfss.logger.clear_messages(proj_name="", des_name="", level=3)
        except Exception as e:
            self.logger.error(f"Failed to clear AEDT messages: {str(e)}")



    def cleanup(self, close_projects = False) -> None:
        """Clean up HFSS resources."""
        hfss = self.hfss
        hfss.save_project()
        hfss.release_desktop(close_projects=close_projects, close_desktop=False)

        self.logger.info("HFSS cleanup completed")


    # Not useful, HFSS will be closed when the object is deleted.
    # def __del__(self) -> None:
    #     """Ensure cleanup on object deletion."""
    #     if self.hfss is not None:
    #         self.hfss.release_desktop(close_projects=False, close_desktop=False)
    #         self.logger.info("HFSS Manager Deleted")


def quick_close_project():
    """
    This function is used to close connection to HFSS project and clean up.
    """
    hfss_manager = HFSSManager()

    # Provides the HFSS application interface.
    # This class allows you to create an interactive instance of HFSS and connect to an existing HFSS design
    # or create a new HFSS design if one does not exist.
    hfss_manager.hfss = ansys.aedt.core.Hfss(
        version=AEDT_VERSION,
        project=None, # Find a default active project
        design=None,  # find a default design in the project
        # solution_type=self.default_settings["solver_type"],
        solution_type=None,
    )
    
    hfss_manager.cleanup()



def testing_function():
    """
    This function is used to test the HFSS manager functions.
    """
    # Test HFSSManager
    hfss_manager = HFSSManager()

    # Start a mew project and design by copying from the template
    # hfss_manager.setup_project_design(
    #     template_filename="RectInset_ATK_10G-template-design.aedt", 
    #     new_project_filename="NewDesign_1.aedt", 
    #     design_solution_type="Terminal"
    #     )

    # Just work on the current active project and design
    hfss_manager.hfss = ansys.aedt.core.Hfss(
        version=AEDT_VERSION,
        project=None, # Find a default active project
        design=None,  # find a default design in the project
        # solution_type=self.default_settings["solver_type"],
        solution_type=None,
    )

    # # Set up simulation analysis setup
    # analysis_params = {
    #     "frequency_range": (24, 25),
    #     "max_passes": 10,
    # }
    # hfss_manager.setup_analysis(analysis_params=analysis_params)

    # hfss_manager.read_design_objects_info()


    # hfss_manager.setup_param_sweeps()

    # hfss_manager.read_design_variables()
    # hfss_manager.read_parameter_sweep()

    # # Add simulation results reports. S11 and S21 for now.
    # hfss_manager.setup_reports()

    # Run the simulation and return the results
    results = hfss_manager.run_simulation()
    print(results)

    results = hfss_manager._gather_results(S_expr_list=["S11", "S21"])

    from modules.design_evaluator import SimResultsEvaluator
    from modules.surface_spec_def import MetasurfaceSpec, PolarizationType, SurfOpMode, SigControlMode, SigControlType
    result_evaluator = SimResultsEvaluator(MetasurfaceSpec(
        reasoning_comment="Testing",
        op_freq_band=0.9,
        op_freq_range=[5.9, 6.1],
        second_freq_band=0,
        second_freq_range=[0, 0],
        bandwidth_ratio=0.1,
        polarization=PolarizationType.LINEAR,
        op_mode=SurfOpMode.TRANS_REFLECT,
        sig_control_mode=SigControlMode.PROGRAMMABLE,
        sig_control_type=SigControlType.AMP,
        sig_control_range="on off",
        deployment_spec=None
    ), results)
    result_report = result_evaluator.parse_results()
    print(result_report)

    # Clean up: disconnect from HFSS.
    hfss_manager.cleanup()


def testing_new_surf_layer():
    """
    This function is used to test the HFSS manager functions.
    """
    # Test HFSSManager
    hfss_manager = HFSSManager()

    # Just work on the current active project and design
    hfss_manager.hfss = ansys.aedt.core.Hfss(
        version=AEDT_VERSION,
        project=None, # Find a default active project
        design=None,  # find a default design in the project
        # solution_type=self.default_settings["solver_type"],
        solution_type=None,
    )

    # Start a mew project and design by copying from the template
    # hfss_manager.setup_project_design(
    #     template_filename=None, 
    #     new_project_filename=None, 
    #     design_solution_type="Modal"
    #     )

    hfss_manager.read_parameter_sweep()

    # hfss_manager.create_metal_sheet("MetalSheet1", ["0mm", "0mm", "-10mm"], ["10mm", "10mm", "0.1mm"])
    # hfss_manager.create_metal_sheet(MetalSheetCreateInfo(
    #     name="MetalSheet1",
    #     position=["0mm", "0mm", "-10mm"],
    #     sizes=["10mm", "10mm", "0.1mm"],
    # ))

    hfss_manager.copy_surf_objects(
        SurfPatternCopyInfo(
            source_obj_list=["SplitRing"],
            move_offset=["0mm", "0mm", "thickness_between_layers"],
            independent_pattern=True,
            independent_circuit=True,
            variables_toCopy=["R1","R2","G_angle","Rw"],
        )
    )

    # hfss_manager.read_design_variables()
    # hfss_manager.read_parameter_sweep()


    # Clean up: disconnect from HFSS.
    hfss_manager.cleanup()



## ======= Example usage and testing code for HFSSManager =====
if __name__ == "__main__":
    
    quick_close_project()

    
    # testing_function()

    # testing_new_surf_layer()
