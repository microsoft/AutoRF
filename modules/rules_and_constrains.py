from pydantic import BaseModel
from enum import Enum
from typing import List


class RuleType(Enum):
    REQ = "requirement, to be followed strictly"
    GUIDE = "guideline, to be followed if possible"
    SUGS = "suggestion, to be considered"


class RuleEntry(BaseModel):
    type: RuleType
    title: str  # short and concise title
    content: str

    def __str__(self):
        return f"**{self.title}**: {self.content}"


class ConstraintType(Enum):
    HARD = "hard constraint, MUST follow"
    SOFT = "soft constraint, best effort follow."


class ConstraintEntry(BaseModel):
    type: ConstraintType
    title: str  # short and concise title
    content: str

    def __str__(self):
        return f"**{self.title}**: {self.content}"


class RulesAndConstrains:
    """Structured specification for design rules."""

    rules: List[RuleEntry]
    constraints: List[ConstraintEntry]

    def __init__(self):
        self.rules = self.get_rules()
        self.tuning_rules = self.get_tuning_rules()
        self.constraints = self.get_constraints()

    def get_common_rules(self) -> List[RuleEntry]:
        """Get the common config rules for both design maker and tuning."""
        rules = [
            RuleEntry(
                type=RuleType.REQ,
                title="Round up numbers",
                content="For scaling factors, round up to the nearest 2 digits number. For example, if the calculated value is `1.12232`, use `1.1`, if `0.8865` use `0.89`. For metasurface pattern dimensions, like radius, length, they also round up to nearest number that is the multiplication of 0.05mm for both nominal design variable and parameter sweeps if possible. For example, `R=0.13mm` should round up to 0.15mm, `R=1.34mm` should round up to 1.35mm for clearness",
            ),
            RuleEntry(
                type=RuleType.GUIDE,
                title="Avoid long simulation",
                content="(1) Do not decrease the step values in sweeps too much or create too many sweep variations, which will cause long simulation time. For example, if the sweep have length, width, gap, each variable having 10 values, then it multiples to 1000 designs, which is too many. Keep the total number of variations in each sweep below 30, for example, width and gap each 6 values, leading to 36 variations. (2) Try to reuse the simulation results from previous design iterations, specifically, updated parameter sweep can overlap with previous sweep to avoid repeated computation. For example, if we have calculated `start=0.1, end=2.0, step=0.1`, and we want to add `step=0.05` and adjust end value, then the new sweep `start=0.1, end=3.0, step=0.05` is better than `start=0.11, end=3.01, step=0.05`, since the later does not reuse or has overlap with previous sweep results. (3) For antenna designs especially, we want extra short simulation, so keep the number of variations below 10, using at most one variable for sweep only! For example, do not sweep over gap and width both 5 values, but only sweep width for 6 values, and maybe try gap in the next iteration. ",
            ),
            RuleEntry(
                type=RuleType.GUIDE,
                title="Substrate Size Adjustment",
                content="The rule of adjusting substrate size is different for metasurface and antenna designs. For metasurfaces, the element or unit cell size is solely dictated by the target frequency, typically, half of the wavelength. It should be adjusted when adapting operating frequency, but should not change when tuning the design's performance at the target frequency. For antennas, the substrate size is flexible, but should be at least larger than the antenna metallic pattern size. When the substrate is not enough to host the port, it causes `port has a non-conducting edge` error as the port on longer touches the ground plane. For both frequency scaling and later tuning, the substrate size should be adjusted accordingly to have enough size to host the metallic pattern.",
            ),
        ]

        return rules

    def get_rules(self) -> List[RuleEntry]:
        rules = [
            RuleEntry(
                type=RuleType.REQ,
                title="Determine Analysis Frequency Range",
                content="Decide the lower and upper bounds for the HFSS simulation frequency range based on the target specifications. First, utilizing your knowledge of frequency bands (e.g., 2.4 ~ 2.5 GHz for 2.4GHz band, 5 ~ 6 GHz for 5GHz band, or 24 ~ 25 GHz for 24GHz mmWave band). Then, expand the range by around 25 percents of the center frequency towards two sides -- [0.75* freq, 1.25*freq](prefer integer number of GHz) so that we can see the wideband performance of the design, for example, 2.4GHz band should be expanded to 2~3 GHz, 5 GHz band should be expanded to 4~7 GHz, 24GHz band should be expanded 18~30GHz.",
            ),
            RuleEntry(
                type=RuleType.REQ,
                title="Identify Variable and Sweep Updates",
                content="Assess which variables and parameter sweeps require updating. Must update auxiliary design variables, including wavelength (lambda), center frequency (freq), according to the current target frequency. For example, when target frequency is 24GHz, we need to update `lambda` to new wavelength value `12.5mm`. Design variables that have an expression that depends on other variables do not need update, such as expression=`lambda / 2`, as they will change automatically. For metal thickness and the circuit capacitance or resistance, do not update or scale them since they are decided by fabrication process and circuit component we use, both of which are fixed.",
            ),
            RuleEntry(
                type=RuleType.REQ,
                title="Scaling for Frequency Band Changes",
                content="Scale all physical dimension-related design variables and sweep values by a scaling factor, new_value = old_value * scaling_factor. Calculate the scaling factor as old_frequency_in_template / new_target_frequency. Make sure the order of these two frequencies is correct! For example, a change from 2.4GHz to 5GHz target frequency means a factor of 2.4 / 5 = 0.48, a change from 60GHz to 30GHz means a factor of 2. Multiply this factor to (1) all independent variables that describes the metallic pattern dimensions, like radius, length, width, gap, and (2) dielectric substrate thickness, unless other limitations are mentioned below. If a parameter sweep is about controlling metasurface geometry, you need to adjust it when frequency scaling is needed. When updating the sweep, you need to adjust the start, end, and step values together, do not forget the step value, all multiplying the factor.",
            ),
            RuleEntry(
                type=RuleType.REQ,
                title="Antenna Matching Structure and Feed Line Scaling",
                content="Not all antenna pattern dimensions need frequency scaling when update operating frequency. 1. For matching structure, you need to recall what's the equation that describes its operation and then decide whether its dimension need scaling for it to work on the target frequency. For example, inset feed can match patch antenna impedance and the inset distance is related to patch length, thus, it needs scaling; edge feed line matches impedance using feed line, whose impedance is decided by substrate mainly, thus, do not need scaling based on frequency. 2. Note that antenna feed line generally does not follow scaling for frequency. Recall related microstrip transmission line theory, the microstrip line dimensions, like width, gap, length, and substrate effective dielectric constant decides the impedance. And we want to maintain the 50 ohms impedance at the target frequency, unless the feed line is an edge line for impedance matching. So we may need to adjust feed line dimensions accordingly when the substrate thickness or kind is changed, instead of applying frequency scaling.",
            ),
            RuleEntry(
                type=RuleType.REQ,
                title="Scaling Substrate Thickness",
                content="MUST scale the thickness or height of substrate layers for frequency and material adjustment, for example, multiply all the FR4 and pvc substrate thickness variables by the factor, unless limited by other constraint.",
                # Given the substrate thickness available options below, we only need a total thickness similar to the sum of scaled substrate thickness. For example, 0.1mm paper and 0.2mm pvc substrate layers should be scaled to 0.25mm paper and 0.1mm pvc, because we do not have the option of 0.25mm paper, it is better to use 0.15mm paper and zero thickness pvc substrate.
            ),
            RuleEntry(
                type=RuleType.REQ,
                title="Adjustments for Material Changes",
                content="Do NOT modify material unless you have to. If the target specification allows ANY material, use the material in the template design. When asked to change substrate material, there are two situations. (1) If there is only one kind of dielectric substrate in the template design, update dielectric substrate material to the demanded one by updating existing substrate. (2) If there are multiple kinds of dielectric substrate used together, create a new substrate with new material, taking the space of those layers precisely, by calculating the summed locations and sizes, then delete all old layers. For example, if the template design uses both paper (size_Z = h_1) and pvc (size_Z = h_2) substrate layers, Update the thickness design variables by multiplying the scaling factor (h_1 = h_1 * scaling_factor, h_2 = h_2 * scaling_factor). Then, delete both paper and pvc substrate layers, and replace with a demanded substrate Box (size_Z = `h_1 +h_2`).\n Note that you need to update paper h_1 and pvc thickness h_2 design variables even though the objects will be deleted later. When specifying the sizes or positions of objects replacing multiple substrate layers, MUST use expression based on existing variables (like Size_Z expression : `h_1 +h_2`) instead of numeric values like `10mm` or `10 * H`, which is to make sure compatibility with other existing objects.",
            ),
            RuleEntry(
                type=RuleType.REQ,
                title="Material Scaling",
                content="When you updated the substrate material to a different kind, recall that the substrate dielectric constant affects the design. You need to multiply another scaling factor for the physical dimension-related design variables, to adjust design variables and sweeps inversely to the square root of the dielectric constant, that means the substrate scaling factor = sqrt( dk_old/ dk_new ). The dielectric constants for the materials are: paper - 3.78, pvc plastic - 2.7, FR4 - 4.4, rogers 4003 - 3.55, rogers 4350 - 3.66, Duroid (tm) - 2.2. ",
                # Note that you should consider the main substrate material, such as from paper+PVC to FR4 needs scaling factor = sqrt( dk_pvc/ dk_fr4 ), while paper+PVC to paper+air should be scaled according to factor = sqrt( dk_pvc/ dk_air ) assuming the paper is very thin relative to operating frequency wavelength.
            ),
        ]

        rules += self.get_common_rules()

        return rules

    def get_constraints(self) -> List[ConstraintEntry]:
        constraints = [
            ConstraintEntry(
                type=ConstraintType.HARD,
                title="Substrate and Design minimal unit",
                content="The thickness of dielectric substrate layer, such as paper, PVC or FR4 or rogers substrate, should be in the unit of 0.1mm only, for example,  0mm, 0.2mm, 0.8mm, 1.0mm. 0mm thickness means you want to remove the substrate, 0.13mm should round up to 0.1mm. The surface pattern dimensions, like radius, length, width, gap, should be greater or equal to 0.1mm and the min value change step/unit is 0.05mm or more, for example, `R=0.1mm, and start=0.1mm, end=2.0mm, step=0.05mm` is valid, `step=0.03mm` is invalid.",
            ),
            ConstraintEntry(
                type=ConstraintType.HARD,
                title="Substrate thickness available options",
                content="We are limited by the fabrication process and material availability. The thickness of the *paper* substrate should be 0.1mm or 0.15mm only, the thickness of FR4 or rogers substrate only have the options of `0.2, 0.3, 0.4`mm and from 0.6 to 3.2mm in the unit of 0.2mm. The thickness of the PVC substrate should be from 0.1mm to 30mm with 0.1mm as the minimal step. \n Do NOT delete substrate object, instead you can set a substrate to zero thickness to remove it if you have to. Do NOT set thickness to zero if desired value is over limit, instead use the closet allowed value!" + 
                "Given the substrate thickness available options, we can make trade-offs between different substrate thickness. For example, if 0.1mm paper and 0.4mm pvc substrate layers should be scaled to 0.025mm paper and 0.1mm pvc, but we do not have the option of 0.025mm paper. Since paper and pvc have similar permittivity, instead of 0.1mm paper and 0.1mm pvc, we should consider them jointly and use 0.15mm paper and 0mm thickness pvc substrate to minimize overall inaccuracy. Note zero thickness is only used when the substrate thickness needs to be very thin, such as when 0.1 mm PVC is not available because thiner than the 0.2mm limit." +
                # Give up air spaver option
                # When the thickness needed is over the available options, such as 10mm PVC, you should update the substrate to air as mentioned in another constraint below, and air has no thickness limit. Make sure that the air substrate still reference thickness variables of previous substrates and update them accordingly by scale to keep surface pattern at the right place! 
                "For antenna designs, specially, we can only use 1.6mm FR4 substrate to match the most common PCB fabrication option and minimize PCB cost and weight!", 
                # Based on Taobao and PCBway/JDB PCB manufacturer
            ),
            #  Must use one of the following material names: `FR4_epoxy`, `Rogers 4003 (tm)`,  `Rogers RO4350 (tm)`, `PVC plastic`, `copper`, `aluminum`.
            ConstraintEntry(
                type=ConstraintType.HARD,
                title="Material Choices",
                content="Must use one of the following material names: `FR4_epoxy`, `Rogers RO4003 (tm)`,  `Rogers RO4350 (tm)`, `PVC plastic`, `polyimide` (for FPC - flexible printed circuit board), `copper`, `aluminum`, `air`. Other material is not supported. Note that paper substrate is often together used with pvc substrate, which is allowed. When target specification mentions paper material only, DO NOT set PVC thickness to 0 or delete it, you can use paper together with PVC!",
                # Note that `air` material is a special case only used when we want a thick substrate unavailable with other materials. For example, when scaling automs surface for 2.4GHz, the thickness of PVC becomes 10mm, above the possible range, we can use update the PVC object to air and set a scaled thickness = 10mm * sqrt(DK_pvc/ DK_air).
            ),
            ConstraintEntry(
                type=ConstraintType.HARD,
                title="HFSS Config constraints",
                content="For parameter sweep in HFSS, Start value of sweep cannot be greater than or equal to stop value when step value is positive. For example, a variable in sweep can not have start value 0.1 and end value 0.1 with step 0.1, which leads to no variation created.",
            ),
            ConstraintEntry(
                type=ConstraintType.HARD,
                title="Programmable Metasurface wires",
                content="Programmable metasurface designs may include thin wires and vias to provide biasing voltages to varactor or PIN diode. The minimum wire width and via radius is 0.1mm due to PCB fabrication capability limits.",
            )
        ]
        return constraints

    def get_tuning_rules(self) -> List[RuleEntry]:
        rules = [
            RuleEntry(
                type=RuleType.GUIDE,
                title="Signal Power Threshold",
                content="For reflective metasurface, we expect the S11 reflection power is higher than -5dB at the target frequency, unless it is amplitude control surface. For transmissive metasurface, we expect the S21 transmission power is higher than -12dB at the target frequency, unless it is amplitude control surface or cross-polarization transmissive surface. Amplitude control surface only needs one variation with high amplitude, and a over 5dB amplitude gap or range. For cross-polarization transmissive surface, however, they are special case and power threshold is higher than -12dB to meet specifications.",
            ),
            RuleEntry(
                type=RuleType.GUIDE,
                title="Signal Control Range",
                content="For phase control metasurface, we need to achieve a phase control range greater than 300 degrees, and a phase resolution/gap smaller than 90 degrees. For amplitude control metasurface, we consider the minimal amplitude control range we need to achieve is 5 dB. We can allow ten percentages up or down for the control range, such as 270 degrees phase range is also allowed, unless a significantly better performing frequency is mentioned.",
            ),
            RuleEntry(
                type=RuleType.GUIDE,
                title="Frequency Scaling Strategy -- Metasurface",
                content="For metasurface specification that current design does not meet, such as not enough phase control range, if the result summary report mentions a better performing frequency, such as a larger phase shift range at a frequency different from the target frequency, you should scale the variables and parameter sweeps related to physical design dimensions (like radius, length, width, substrate thickness) by a scaling factor = good_operating_frequency / target_operating_frequency to tune it to achieve the target specifications. For instance, if the target frequency is 10 GHz with 180 degrees phase range, and we see a 290 deg phase control range at 5GHz operating frequency, you should scale the design by a factor of 0.5.  The better performing frequency does not need to meet target spec, it only needs to be over 50% better than current and relatively close to our target signal control range. For example, from 100 at target frequency to 200 at the other frequency is worth scaling, while from 10 at target frequency to 50 at the other frequency is not worth scaling. When updating the sweep, you need to adjust the start, end, and step values together, do not forget the step value, proportionally to the frequency change. MUST scale the thickness or height of all substrate layers too, for example, scale all the FR4 and pvc substrate thickness by the factor! Make sure that you do NOT scale the total width of the surface substrate or the spacing of surface element or the thickness of metal layers, which is already decided and fixed by the target frequency or the fabrication process.",
            ),
            RuleEntry(
                type=RuleType.GUIDE,
                title="Frequency Scaling Strategy -- Antenna",
                content="For antenna specification that current design does not meet, such as not enough S11, if the result summary report mentions a better performing frequency, such as lower than -10dB S11 at a frequency different from the target frequency, you should scale the variables related to physical design dimensions (like length, width) by a scaling factor = good_operating_frequency / target_operating_frequency to tune it to achieve the target specifications. For example, the target frequency is 5GHz while a better frequency is observed at 3.5GHz, you can scale the antenna dimensions by 3.5/ 5 = 0.7. You do not need to scale the feed line dimensions, as they follow transmission line theory that do not scale with frequency. You do not need to scale the thickness or height of substrate layers for antennas, since they are typically decided by PCB fabrication process constraints.",
            ),
            RuleEntry(
                type=RuleType.GUIDE,
                title="Tuning Exploration Strategies",
                content="On the other hand, if no better performing frequency is reported, you should explore the design space by consider what are the possible ways to tune the design variables and parameter sweeps for achieving the target spec. Potential strategies include: (1) Reduce the step value of one variable in the sweep to increase number of variations, so that we sample the search space in finer granularity. If see improvement, can repeat with smaller steps. (2) Increase the range of design parameter sweeps within the max range that does not cause error. For example, for a element spacing (width of whole element) of 1mm, the outer radius of a ring pattern should not exceed 0.5mm.  (3) Adjust one or multiple nominal design variable values, not by a same scaling factor but according to your knowledge of what can help achieve a specification match, to explore a better design. For example, tuning the substrate thickness, change pattern width. But do NOT add substrate thickness to parameter sweep, as mentioned in another rule. (You should provide a detailed explanation for any decision you made.)",
            ),
            RuleEntry(
                type=RuleType.GUIDE,
                title="Valid Variable for Sweep",
                content="You can explore the following kinds of design variables for the sweeps: For passive non-programmable metasurfaces, you can explore metasurface pattern dimensions related variables, such as the length, width, gap, radius. For programmable metasurface, you can only update nominal design variable value, that means you can NOT sweep the value of circuit components, such as resistance and capacitance, as they are pre-decided. Do NOT add pattern dimension variables, such as patch width, in sweep for programmable metasurfaces which only use circuit components as variations. Do NOT add variables that we can not control individually for each element, for example, do NOT add substrate thickness in parameter sweep since we can NOT control signal by changing substrate thickness for each element.",
            ),
            RuleEntry(
                type=RuleType.GUIDE,
                title="Do Not Break the Design",
                content="Do not alter design variables that already have an expression to define them. As they are defined by other variables, they are automatically updated and not independent. For example, if the variable `R2` is defined as `R1- Rw`, you should not change it to `1mm`. Do not update variables or sweep values that are about rotate angles,or have `angle` in the variable name, as they are pre-defined. Do NOT change or update the parameter sweep, when the current sweep uses circuit component values, such as resistance, capacitance, as the values are fixed and decided by the circuit design.",
                #  You should not change an independent variable to an dependent variable, for example, Rw=0.5mm can be changed to 0.1mm but should not be updated to `Rw`=R1-R2. (For gpt-4o only, strange mistake.)
            ),
            RuleEntry(
                type=RuleType.GUIDE,
                title="Antenna impedance matching",
                content="For antenna design tuning, you should consider tuning the impedance. If S11 at target frequency is high and no better frequency nearby -- deviates from the target frequency within 40%, you can try adjusting the antenna impedance matching structure first, such as edge feed width, inset width or gap, to improve the S11 at the target frequency or nearby frequencies and then scaling the operating frequency in the next iteration. ",
            )
        ]

        rules += self.get_common_rules()

        return rules


    def _format_rules(self, rules: List[RuleEntry]) -> str:
        """Format the rules into a string for display."""
        rule_str = ""
        for id, rule in enumerate(rules):
            rule_str += str(id + 1) + "." + str(rule) + "\n"
        return rule_str

    def rules_str(self):
        return self._format_rules(self.rules)

    def constraints_str(self):
        return self._format_rules(self.constraints)

    def tuning_rules_str(self):
        return self._format_rules(self.tuning_rules)

    def __str__(self):
        return f"Rules: \n{self.rules_str()} \nConstraints: \n{self.constraints_str()}"


if __name__ == "__main__":
    rules = RulesAndConstrains()
    print(rules)
