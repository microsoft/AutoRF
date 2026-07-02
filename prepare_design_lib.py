if __name__ == "__main__":
    import sys, os

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, asdict
from pydantic import BaseModel

from modules.surface_spec_def import MetasurfaceSpec, AntennaSpec
from modules.utils.custom_logger import setup_logger

from modules.utils.pdf_extraction import PDFExtractor

from modules.literature_learner import DesignLitSummary, LitratureLearner


def add_metasurface_template(paper_list: list):
    """
    Add a metasurface template to the design library.
    """
    # Initialize the LLM interface
    from modules.utils.llm_interface import OpenAILLMInterface
    from modules.utils.yaml_config_manager import YAMLConfigManager

    llm_interface = OpenAILLMInterface()

    # Example usage
    lit_learner = LitratureLearner(llm_interface)

    for paper in paper_list:
        design_summary = lit_learner.summarize_lit(paper)

        import pickle

        # dump the design summary to the design library folder as a object
        with open(
            f"./design_library/design_summary/{design_summary.nickname}.pkl", "wb"
        ) as file:
            pickle.dump(design_summary, file)

        # # save the design summary to the design library folder as a yaml file
        # YAMLConfigManager().save_config(asdict(design_summary), "{design_summary.nickname}")

        # print(design_summary)


def add_antenna_template(antenna_summary: DesignLitSummary):
    """
    Add a antenna template to the design library.
    """

    import pickle

    # dump the design summary to the design library folder as a object
    with open(
        f"./design_library/design_summary/{antenna_summary.nickname}.pkl", "wb"
    ) as file:
        pickle.dump(antenna_summary, file)

    print(antenna_summary)
    print("Added antenna template to the design library.")


if __name__ == "__main__":

    # Add a metasurface template
    # add_metasurface_template(["./design_library/paper_pdf/automs-final.pdf",])

    # Add a antenna template
    antenna_summary = DesignLitSummary(
        nickname="RectInset_ATK_10G", # Should be the same as the HFSS design file name prefix. "AutoMS-template-design.aedt" should put "AutoMS" here.
        application_scenario="generic antenna design",
        design_description="A rectangular inset-fed patch antenna design for 10 GHz operation.",
        design_params="See the HFSS design variables.",
        hardware_spec=AntennaSpec(
            reasoning_comment="Skip for manual input",
            op_freq_band=10, # GHz
            op_freq_range=[10.6, 10.95], # range that S11 < -10 dB, patch antenna has narrow bandwidth.
            polarization="linear",
        ),
    )
    add_antenna_template(antenna_summary)

    antenna_summary = DesignLitSummary(
        nickname="EllipticalEdge_ATK_10G", # Should be the same as the HFSS design file name prefix. "AutoMS-template-design.aedt" should put "AutoMS" here.
        application_scenario="generic antenna design",
        design_description="A Elliptical edge-fed (narrow edge feed line for impedance matching) patch antenna design for 10 GHz operation.",
        design_params="See the HFSS design variables.",
        hardware_spec=AntennaSpec(
            reasoning_comment="Skip for manual input",
            op_freq_band=11.1, # GHz
            op_freq_range=[10.8, 11.5], # range that S11 < -10 dB, patch antenna has narrow bandwidth.
            polarization="linear",
        ),
    )
    
    add_antenna_template(antenna_summary)
