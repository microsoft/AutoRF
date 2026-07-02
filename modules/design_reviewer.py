if __name__ == "__main__":
    import sys, os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils.custom_logger import setup_logger
from modules.utils.custom_logger import Colors
from modules.utils.llm_interface import prompt_dict_to_string


from modules.design_info_def import MacroDesignDecision, DesignInfo


from dataclasses import dataclass, asdict
from pydantic import BaseModel
from enum import Enum
from typing import List


# class ReviewDecision(Enum):
#     PASSED = "Review passed, GO ahead with the design edits"
#     # NO_GO = "Review failed, NO GO for the design edits, need re-iteration"
#     FAILED_MISS_RULE = "Review failed to pass, design does not follow general design rules."
#     FAILED_CONSTRAINTS = "Review failed, design violates the constraints of practical implementation."


class DesignReviewerFeedback(BaseModel):
    """Decision of the design reviewer with LLM interface"""

    reasoning_comment: str
    passed: bool  # True for success, False for failure


class DesignReviewer:
    """Evaluate the results of the design and decide further tuning step with LLM interface"""

    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.logger = setup_logger()

        # self.output_schema = DesignReviewerFeedback

    def review_design_decision(
        self, messages: List[dict]
    ) -> DesignReviewerFeedback:
        """
        Evaluate the results of the design with LLM interface and return the tuning decision.

        """
        try:
            self.logger.info("Starting design reviewer...")

            messages.append(
                {
                    "role": "user",
                    "content": """Read the background information in previous messages. Review the json formatted macro design edits another assistant just proposed above. Specifically, check against the following aspects: 
1. Does the design follow the general design rules? Some common mistakes: 
    (1) design variables or sweeps calculated incorrectly. For example, you should NOT set PVC thickness to 0 because it is allowed to exist and work together with paper substrate. You should NOT edit sweep values of circuit components such as resistance, capacitance, because this is not allowed as circuit components are fixed. You should not create a new substrate with a explicit numerical thickness value, like `3mm` or `6 * H`.
    (2) a design variable got forgotten to be scaled or updated, such as wavelength lambda, pattern width. The design may mentioned certain variables, but eventually forgot to include in the final summary or output explicitly. But it is correct to skip variables in some cases, such as when lambda is not in the design file then we don't need to update it.
    (3) Unnecessary design edits, such as updating substrate material even when the target spec allows any material and current material satisfies design constraints. On the other hand, updating substrate thickness for frequency scaling or updating substrate material to FR4 due to practical material availability is valid and necessary. 
    (4) Forgot to apply the material scaling factor to related design variables when updating the substrate material, for example, from paper to FR4.
2. Does the design violate any constraints of practical implementation? Common cases include: (1) the thickness of FR4 substrate is thicker than practical available options.
3. Does the design object edits violate 3D placement? Common cases include: 
    (1) edit unrelated or unnecessary design objects. 
    (2) object edit using specific numbers instead of expression of existing design variables, for example, should use `h_pvc+h` instead of 1mm, which may cause errors when other design dimensions change. The exception is that 0mm or 0 is allowed. 
    (3) New objects intersects with other existing objects. 
If any check above fails, provide a detailed explanation and suggestions for improvement and put `REJECT` at the end your response with no other characters after it. If the design is good, say `ACCEPT` at the end of your response.""",
                }
            )

            self.logger.debug(
                f"==== Added User Message ==== \n {messages[-1]["content"]}"
            )

            # Call LLM for translation
            response = self.llm.get_string_response(messages)
            # self.logger.debug(f"LLM response: {response}")

            if "ACCEPT" in response[-10:]: # check the last 10 characters for ACCEPT or REJECT
                feedback = DesignReviewerFeedback(reasoning_comment=response, passed=True)
                self.logger.info("Design review says accepted!")
            elif "REJECT" in response[-10:]: # check the last 10 characters for ACCEPT or REJECT
                feedback = DesignReviewerFeedback(reasoning_comment=response, passed=False)
                self.logger.info("Design review says rejected!")
            elif "ACCEPT" in response[-100:]:
                feedback = DesignReviewerFeedback(reasoning_comment=response, passed=True)
                self.logger.warning("Design review says accepted! but the format is wrong.")
            elif "REJECT" in response[-100:]:
                feedback = DesignReviewerFeedback(reasoning_comment=response, passed=False)
                self.logger.warning("Design review says rejected! but the format is wrong.")
            else:
                feedback = DesignReviewerFeedback(reasoning_comment=response, passed=False)
                self.logger.error("Design review unknown response! Treated as rejected.")
                # assert False, f"Unexpected response from LLM: {response}"
                
            self.logger.debug(f"LLM response: {response}")      

            # Pop out the last message since we do not want to include the prompt here for the design maker. They share the messages!
            self.logger.debug(f"Popped out last message.")
            messages.pop(-1)

            self.logger.info("Design Reviewer completed successfully")

            return feedback

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
    demand_translator = DesignReviewer(llm_interface)
