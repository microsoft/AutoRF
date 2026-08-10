from pydantic import BaseModel
from dataclasses import dataclass
from enum import Enum

class SurfOpMode(Enum):
    TRANS = "transmissive"
    REFLECT = "reflective"
    TRANS_REFLECT = "both transmissive and reflective." # e.g., for power on and off amplitude control
    ABSORB = "absorptive"

class SigControlMode(Enum):
    PROGRAMMABLE = "programmable/reconfigurable"
    PASSIVE = "passive/fixed"


class SigControlType(Enum):
    PHASE = "phase"
    AMP = "amplitude"
    POLAR = "polarization"
    IMPEDANCE = "impedance"


class PolarizationType(Enum):
    ALL_ALLOWED = "all allowed, no specific requirement"
    LINEAR = "linear"
    CIRCULAR = "circular"
    CROSS = "cross-polarization, converting from one to another" # Do we consider polarization conversion from metasurface?


class MetasurfaceMaterial(Enum):
    METAL = "metal"
    FR4 = "fr4"
    ROGERS = "rogers"
    Paper = "paper"
    FPC = "FPC"
    ANY = "any available material"

class DeploymentSpec(BaseModel):
    """Structured specification for metasurface deployment constraints, like num of layers, material, thickness."""
    material: MetasurfaceMaterial
    max_num_of_layers: int
    max_thickness_allowed_mm: float
    max_cost_per_element: float

# @dataclass
class MetasurfaceSpec(BaseModel):
    """Structured specification for metasurface design."""
    ## 0. LLM Reasoning for the design
    reasoning_comment: str 
    ## 1. basic characteristics
    # Op frequency range in GHz
    op_freq_band: float
    op_freq_range: list[float] 
    second_freq_band: float
    second_freq_range: list[float] 
    bandwidth_ratio: float
    polarization: PolarizationType

    # Surface operation modes
    op_mode: SurfOpMode 
    min_power_efficiency_dB: float # minimum power efficiency in dB, e.g., -10dB, to capture power loss.
    ## 2. Signal control
    sig_control_mode: SigControlMode
    sig_control_type: SigControlType
    sig_control_range: str # e.g., "0 to 360 degrees", "10dB amplitude difference"
    # tuning_range: list[(float, float)]  # min, max
    ## 3. deployment constraints
    deployment_spec: DeploymentSpec #| None
    # material: str
    # additional_constraints: Dict[str, Any]

class AntennaSpec(BaseModel):
    """Structured specification for antenna design."""
    ## 0. LLM Reasoning for the design
    reasoning_comment: str 
    ## 1. basic characteristics
    # Op frequency range in GHz
    op_freq_band: float
    op_freq_range: list[float] | None = None
    second_freq_band: float | None = None
    second_freq_range: list[float] | None = None
    polarization: PolarizationType

class HardwareType(Enum):
    METASURFACE = "metasurface"
    ANTENNA = "antenna"

class HardwareTypeChoice(BaseModel):
    hardware_type: HardwareType
