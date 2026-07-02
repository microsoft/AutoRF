from pydantic import BaseModel
from typing import List
from dataclasses import dataclass
from enum import Enum

if __name__ == "__main__":
    import sys, os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.hfss_manager import VariablePropertyInfo, VariableSweepInfo
from modules.hfss_manager import ObjectPropertyInfo, MaterialEditInfo, BoxCreateInfo, ObjDeleteInfo
from modules.hfss_manager import MetalSheetCreateInfo, SurfPatternCopyInfo




class DesignStrategy(Enum):
    PERFECT_MATCH = "perfect match already, no tuning needed"

    FREQ_SCALE = "frequency scaling"
    MATERIAL_CHANGE = "substrate material change"
    OP_MODE_SWITCH = "switch between transmissive and reflective mode"
    IMPEDANCE_MATCH = "Adjust metasurface impedance to match with a new operation environment."
    BANDWIDTH_TUNE = "bandwidth tuning"
    DUPLI_LAYERS = "Use extra duplicate layers to increase signal control range."

    # NEW_DEF = "Define a novel strategy, described in strategy_comment"
    IMPOSSIBLE = "impossible to meet the specification"

    # PARAM_TUNE = "runing parameters for trade-off"
    # REORG_LAYERS = "reorganize layers" # TODO: support mixing design layers.

    @classmethod
    def print_list(self):
        formatted = "\n".join(f"- `{strategy.name}`: {strategy.value}" for strategy in DesignStrategy)
        return formatted
    


# @dataclass
class DesignProposal(BaseModel):
    """Structured design proposal for metasurface design."""
    reasoning_comment: str # LLM Reasoning for the design
    # matching_score: float # Matching score of the proposed design
    
    template_name: str # Name of the design template
    mismatched_spec: List[str] # List of mismatched specifications for later tuning

    design_strategies: List[str] # Proposed overarching design strategy
    strategy_comment: str # Comment on the chosen design strategy

    circuit_model_blocks: str # High-level circuit model for the proposed design
    new_design_variables: List[str] # List of design variables when adding new layers




# @dataclass
class DesignInfo(BaseModel):
    """Dataclass for storing HFSS design information."""
    design_id: int
    design_filename : str
    var_info_list: List[VariablePropertyInfo]
    sweep_info_list: List[VariableSweepInfo]
    object_info_list: List[ObjectPropertyInfo]

class MacroDesignDecision(BaseModel):
    """Decision made during macro design reasoning by LLM."""
    reasoning_comment: str
    calculation_process: str
    analysis_freq_range: List[float]
    # size_scaling: float
    # material_update: str
    # layer_stacking: List[str]
    variable_updates: List[VariablePropertyInfo]
    sweep_updates: List[VariableSweepInfo]
    material_update: List[MaterialEditInfo]
    object3D_delete: List[ObjDeleteInfo]
    object3D_create: List[BoxCreateInfo]

    ## For new surface design.
    variable_creates: List[VariablePropertyInfo] | None
    metal_create: MetalSheetCreateInfo| None
    surf_copy: SurfPatternCopyInfo| None # default None is not permitted by openAI structured outputs
    


if __name__ == "__main__":
# print out the json format schema
    print(MacroDesignDecision.model_json_schema())
    print(MacroDesignDecision.__get_pydantic_json_schema__())

