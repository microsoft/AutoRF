from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from dataclasses import dataclass
import json
import numpy as np
from io import StringIO
import sys

if __name__ == "__main__":
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.surface_spec_def import *
from modules.hfss_manager import SimulationResults, SimFigureResults
# from modules.utils.llm_interface import OpenAILLMInterface

from modules.utils.custom_logger import setup_logger


class EvaluationCriteria(BaseModel):
    """
    Numeric thresholds used to judge simulation results.

    Values default to the original hard-coded constants, so behaviour is
    unchanged unless a threshold is explicitly overridden or derived from the
    target specification via ``from_spec``.
    """
    # Metasurface magnitude thresholds (dB)
    mag_low_thresh: float = -3.0             # S-parameter magnitude floor for low loss
    cross_pol_mag_low_thresh: float = -10.0  # more loss allowed for cross-polarization
    # Metasurface phase threshold (degrees)
    phase_range_threshold: float = 270.0
    # Antenna magnitude thresholds (dB)
    antenna_s11_thresh: float = -10.0
    antenna_trough_thresh: float = -5.0
    # Frequency matching tolerance (GHz)
    freq_tolerance_ghz: float = 0.5
    # Offsets used when searching for a better frequency point (degrees)
    better_phase_abs_offset: float = 180.0
    better_phase_min_improvement: float = 30.0

    @classmethod
    def from_spec(cls, target_spec) -> "EvaluationCriteria":
        """
        Build criteria from the target spec, falling back to the defaults above.

        Currently derives the low-loss magnitude floor from the demanded power
        efficiency budget (``min_power_efficiency_dB``) when a valid one is given.
        """
        criteria = cls()
        min_eff = getattr(target_spec, "min_power_efficiency_dB", None)
        if isinstance(min_eff, (int, float)) and -30.0 <= min_eff < 0.0:
            criteria.mag_low_thresh = float(min_eff)
        return criteria


class SimResultsEvaluator():
    """Evaluate the simulation results and decide whether the design is successful or not."""
    
    def __init__(self, target_spec: MetasurfaceSpec, results: SimulationResults, criteria: "EvaluationCriteria" = None):
        
        self.logger = setup_logger()
        self.output_schema = dict

        self.logger.debug(f"Input results: {results}")

        self.target_spec = target_spec
        self.results = results
        self.criteria = criteria if criteria is not None else EvaluationCriteria.from_spec(target_spec)
    
    def parse_results(self) -> str:
        """
        Parse the results of the design to return more representative metrics for the LLM to reason about the design.
        """

        if self.results is None or self.results.s_parameters is None:
            return "No simulation results available!"
        
        if isinstance(self.target_spec, MetasurfaceSpec):
            # Evaluate metasurface designs according to the Operation mode.
            if self.target_spec.op_mode == SurfOpMode.REFLECT and self.target_spec.sig_control_type == SigControlType.PHASE:
                report = self.parse_surf_mag("S11") + "\n" + self.parse_surf_phase("S11")
            elif self.target_spec.op_mode == SurfOpMode.TRANS and self.target_spec.sig_control_type == SigControlType.PHASE:
                report = self.parse_surf_mag("S21") + "\n" + self.parse_surf_phase("S21")
            elif self.target_spec.sig_control_type == SigControlType.AMP:
                report = self.parse_surf_mag("S11") + self.parse_surf_mag("S21")
            else:
                # Gracefully report unsupported combinations instead of crashing.
                self.logger.warning(
                    f"Unsupported metasurface evaluation combination: op_mode={self.target_spec.op_mode}, "
                    f"sig_control_type={self.target_spec.sig_control_type}."
                )
                return (
                    "Automated evaluation is not available for this combination "
                    f"(op_mode={self.target_spec.op_mode.value}, control_type={self.target_spec.sig_control_type.value})."
                )
        elif isinstance(self.target_spec, AntennaSpec):
            # Get the report for Antenna S11 evaluation
            report = self.parse_antenna_S11_mag()
        else:
            self.logger.warning(f"Unsupported design type for evaluation: {type(self.target_spec).__name__}.")
            return f"Automated evaluation is not available for design type {type(self.target_spec).__name__}."
        
        return report
    

    def parse_surf_mag(self, S_expr) -> str:
        """Parse the results of the reflective metasurface."""

        S11_low_thresh = self.criteria.cross_pol_mag_low_thresh if self.target_spec.polarization == PolarizationType.CROSS else self.criteria.mag_low_thresh # More loss allowed for cross-polarization

        # 1. Check S21 is low.

        # 2. Check S11 is high at the target frequency and calculate the bandwidth.
        S11_result = self.results.s_parameters[f"{S_expr} Mag"]
        assert S11_result.primary_sweep == "Freq", "S11 result is not a frequency sweep."

        frequencies = np.array(S11_result.primary_sweep_values)
        s11_values = np.array(S11_result.data_lines)

        # find the index of the entry closest to the target frequency
        target_index = np.argmin(np.abs(frequencies - self.target_spec.op_freq_band))
        assert abs(frequencies[target_index] - self.target_spec.op_freq_band) < self.criteria.freq_tolerance_ghz, "Target frequency is not in the sweep range."

          # Extract S11 values at the target frequency across variations
        s11_at_target = s11_values[:, target_index]
        
        # Calculate max, min, and avg values at the target frequency
        max_at_target = np.max(s11_at_target)
        min_at_target = np.min(s11_at_target)
        avg_at_target = np.mean(s11_at_target)

        amp_range = np.max(s11_values, axis=0) - np.min(s11_values, axis=0)
        

        # Determine the max frequency range around the target frequency above -3 dB
        above_threshold = s11_values > S11_low_thresh  # Boolean array
        range_indices = np.where(above_threshold[:, target_index])[0]

        if range_indices.size > 0:
            left_idx = right_idx = target_index
            while left_idx > 0 and np.all(s11_values[:, left_idx - 1] > S11_low_thresh):
                left_idx -= 1
            while right_idx < len(frequencies) - 1 and np.all(s11_values[:, right_idx + 1] > S11_low_thresh):
                right_idx += 1
            frequency_range = (frequencies[left_idx] , frequencies[right_idx])
        else:
            frequency_range = None

        # Store the frequency range for later use as a mask for phase analysis
        self.S11_mag_values = s11_values
        self.S11_high_mag_mask = np.min(s11_values, axis=0) > S11_low_thresh


        # Check if there exists a frequency with higher min and avg values than at the target
        # min_values = np.min(s11_values, axis=0)
        avg_values = np.mean(s11_values, axis=0)
        higher_avg_freq = frequencies[np.argmax(avg_values)] # Find the index with highest S11 avg

        # Prepare the results
        result_report = (
            f"{S_expr} magnitude report:\n"
            f"At target frequency, {S_expr} magnitude max = {max_at_target:.2f}dB, min = {min_at_target:.2f}dB, avg = {avg_at_target:.2f}dB.\n"
            f"For amplitude control surface, that means the amplitude control range is {(max_at_target - min_at_target) :.2f} dB. \n"+
            f"For phase control surface, frequency range with high {S_expr} magnitude: "+
            (f"{frequency_range[0]:.2f} GHz to {frequency_range[1]:.2f} GHz.\n" if frequency_range is not None else "None.\n")
            # f"There exists frequency with higher average {S_expr} magnitude values at {higher_avg_freq:.2f} GHz."
        )

        return result_report

    def parse_surf_phase(self, S_expr: str) -> str:
        """
        Analyze the phase gap and frequency range for metasurface designs.
        """

        S11_result = self.results.s_parameters[f"{S_expr} Phase"]
        assert S11_result.primary_sweep == "Freq", "S11 result is not a frequency sweep."

        frequencies = np.array(S11_result.primary_sweep_values)
        s11_phases = np.array(S11_result.data_lines)

        # assert len(s11_phases) > 1, "No enough phase data available for analysis! Wrong HFSS config?"
        if len(s11_phases) <= 1:
            return "No enough phase data available for phase analysis! Wrong HFSS config or wrong sweep parameter values?"

        # phase_gap_threshold = 60 # degrees, adjust as needed
        phase_range_threshold = self.criteria.phase_range_threshold # degrees, adjust as needed

        target_freq = self.target_spec.op_freq_band

        # Find the index of the target frequency
        target_idx = np.argmin(np.abs(frequencies - target_freq))

        ## 1. Get the max range of phase change across the variations.
        # === Wrong calculation below... Should not unwrap, phase always in [0, 360) degrees. === 
        # unwrapped_phases = np.unwrap(s11_phases, axis=1, period=360) # Unwrap phase values to avoid discontinuities
        # phase_ranges = np.max(unwrapped_phases, axis=0) - np.min(unwrapped_phases, axis=0)
        # phase_range_at_target = phase_ranges[target_idx]

        ## 2. Get the max phase gaps (resolution) among the variations.
        # Ensure the phase values are wrapped to [0, 360) degrees
        wrapped_phases = np.mod(s11_phases, 360)
        sorted_phases = np.sort(wrapped_phases, axis=0) # from smallest to largest phase values

        # get the sorted index list for the phase values at the target frequency
        sorted_idx = np.argsort(wrapped_phases[:, target_idx])
        # get the variation list corresponding to the sorted index
        sorted_variations_at_target = [S11_result.variations[x] for x in sorted_idx]

        # import matplotlib.pyplot as plt
        # plt.plot(sorted_phases[::10, :].T)
        # plt.show()

        # Calculate phase differences
        phase_diffs = np.diff(sorted_phases, axis=0, prepend=sorted_phases[-1:, :] - 360) # IMPORTANT: minus 360 to the last row to get the cyclic difference!
        assert np.all(phase_diffs >= 0), "Phase differences should be positive!"
        # phase_diffs = np.minimum(phase_diffs, 360 - phase_diffs) # Ensure the phase differences are within [0, 180] degrees
        phase_gaps = np.max(phase_diffs, axis=0)

        # get the index of the largest phase gap at the target frequency
        max_gap_idx = np.argmax(phase_diffs[:, target_idx])
        lower_end_index = np.mod(max_gap_idx - 1, len(sorted_variations_at_target))
        

        # import matplotlib.pyplot as plt
        # plt.plot(phase_gaps)
        # plt.show()

        phase_ranges = 360 - phase_gaps
        phase_range_at_target = phase_ranges[target_idx]
        
        # resolution is the second largest gap
        phase_resolutions = np.sort(phase_diffs, axis=0)[-2]
        # Calculate the phase resolution at the target frequency
        phase_resolution_at_target = phase_resolutions[target_idx]

        ## 3. Get the bandwidth where the phase range is large enough
        
        # Boolean mask where phase differences are below the threshold
        above_threshold = phase_ranges > phase_range_threshold

        # Find the max frequency range where the phase range is above the threshold
        max_range = None
        if np.any(above_threshold) and above_threshold[target_idx]:
            left_idx = target_idx
            right_idx = target_idx
            while left_idx > 0 and above_threshold[left_idx - 1]:
                left_idx -= 1
            while right_idx < len(frequencies) - 1 and above_threshold[right_idx + 1]:
                right_idx += 1
            
            max_range = (frequencies[left_idx], frequencies[right_idx])

        ## 4. Get suggested better frequency point for scaling.

        # (1) Apply a mask based on S11 high magnitude frequency range
        phase_ranges_masked = np.where(self.S11_high_mag_mask, phase_ranges, 0)

        # (2) Check if a better phase gap exists at other frequencies (enough phase change and minimal S11 loss)
        if phase_range_at_target < phase_range_threshold:
            # set another mask: must larger than range threshold OR much better (180 deg) than the target phase gap.
            good_phase_mask = (phase_ranges_masked > phase_range_threshold) | (phase_ranges_masked > phase_range_at_target + self.criteria.better_phase_abs_offset)
            # find the one with the highest S11 mag values in the masked range
            S11_mag_min = np.min(self.S11_mag_values, axis=0)
            # Find the index with highest S11 mag
            better_gap_idx = np.argmax( np.where(S11_mag_min, good_phase_mask, -np.inf)) 
            # IF better than 30 degrees lager, we say it is a better gap that worth tuning for!
            if phase_ranges_masked[better_gap_idx] > phase_range_at_target + self.criteria.better_phase_min_improvement: 
                better_gap_exists = (phase_ranges_masked[better_gap_idx], frequencies[better_gap_idx])
            else:
                better_gap_exists = (None, None)
        else:
            better_gap_exists = (None, None)


        # Prepare the results
        result_report = (
            f"{S_expr} phase report:\n" +
            f"At target frequency, {S_expr} phase range = {phase_range_at_target:.2f} degrees and phase resolution = {phase_resolution_at_target:.2f} degrees\n" +
            # f"The variations at two ends of the phase range are {sorted_variations_at_target[lower_end_index]} ({sorted_phases[lower_end_index, target_idx]} deg) and {sorted_variations_at_target[max_gap_idx]} ({sorted_phases[max_gap_idx, target_idx]} deg).\n" +
            f"Frequency range with at least {phase_range_threshold} degrees phase control range = "+
            (f"{max_range[0]:.2f} GHz to {max_range[1]:.2f} GHz.\n" if max_range is not None else "None.\n") + 
            f"There exists frequency with a larger phase range: "+ 
            (f"{better_gap_exists[0]} deg at {better_gap_exists[1]} GHz." if better_gap_exists[0] is not None else "None")
        )

        return result_report
        # return {
        #     'phase_gap_at_target': phase_gap_at_target,
        #     'max_freq_range': max_range,
        #     'smaller_gap_exists': smaller_gap_exists
        # }

    def parse_antenna_S11_mag(self) -> str:
        """Parse the results of the antenna design S11."""

        S11_low_thresh = self.criteria.antenna_s11_thresh

        # 2. Check S11 is high at the target frequency and calculate the bandwidth.
        S11_result = self.results.s_parameters["S11 Mag"]
        assert S11_result.primary_sweep == "Freq", "S11 result is not a frequency sweep."

        frequencies = np.array(S11_result.primary_sweep_values)
        s11_values = np.array(S11_result.data_lines)
        # assert s11_values.ndim == 1, "S11 data is not 1D array."
        if s11_values.ndim == 1:
            # convert to 2D array for consistency
            s11_values = s11_values.reshape(1, -1)
        elif s11_values.ndim == 2:
            # if multiple variations, choose only the one with lowest S11 magnitude value at target frequency
            min_s11_idx = np.argmin(s11_values[:, np.argmin(np.abs(frequencies - self.target_spec.op_freq_band))])
            s11_values = s11_values[min_s11_idx, :].reshape(1, -1)
        else:
            assert False, "S11 data is not 1D or 2D array."

        # find the index of the entry closest to the target frequency
        target_index = np.argmin(np.abs(frequencies - self.target_spec.op_freq_band))
        if abs(frequencies[target_index] - self.target_spec.op_freq_band) >= self.criteria.freq_tolerance_ghz:
            self.logger.error(f"Target frequency is not in the sweep range! Target freq: {self.target_spec.op_freq_band}, closest freq: {frequencies[target_index]}")
            return "Target frequency is not in the sweep range!"
        

          # Extract S11 values at the target frequency across variations
        s11_at_target = s11_values[:, target_index]
        
        # Calculate max, min, and avg values at the target frequency
        max_at_target = np.max(s11_at_target)
        min_at_target = np.min(s11_at_target)
        avg_at_target = np.mean(s11_at_target)

        # Determine the max frequency range around the target frequency above -3 dB
        above_threshold = s11_values <= S11_low_thresh  # Boolean array
        range_indices = np.where(above_threshold[:, target_index])[0]

        if range_indices.size > 0:
            left_idx = right_idx = target_index
            while left_idx > 0 and np.all(s11_values[:, left_idx - 1] <= S11_low_thresh):
                left_idx -= 1
            while right_idx < len(frequencies) - 1 and np.all(s11_values[:, right_idx + 1] <= S11_low_thresh):
                right_idx += 1
            frequency_range = (frequencies[left_idx] , frequencies[right_idx])
        else:
            frequency_range = None

        # Store the frequency range for later use as a mask for phase analysis
        self.S11_mag_values = s11_values
        self.S11_high_mag_mask = np.min(s11_values, axis=0) > S11_low_thresh


        # Check if there exists a frequency with better values than at the target
        # min_values = np.min(s11_values, axis=0)
        avg_values = np.mean(s11_values, axis=0)
        # find the index for all the troughs in avg_values, each trough is a potential better frequency. 
        # trough is local minimum so that it is smaller than its neighbors
        better_freq_mask = np.r_[False, avg_values[1:] < avg_values[:-1]] & np.r_[avg_values[:-1] < avg_values[1:], False]
        # further clean up the mask by removing the ones with S11 values above -3 dB
        better_freq_mask = better_freq_mask & (avg_values < self.criteria.antenna_trough_thresh)
        better_freq_list = frequencies[better_freq_mask]
        better_s11_values = avg_values[better_freq_mask]

        # Prepare the results
        result_report = (
            "S11 magnitude report:\n" +
            (f"We only report the variation that has the lowest S11 magnitude values at target frequency, variation: {S11_result.variations[min_s11_idx]}]\n" if s11_values.ndim == 2 and len(S11_result.variations)>0 else "") +
            f"At target frequency, S11 magnitude max = {max_at_target:.2f}dB, min = {min_at_target:.2f}dB, avg = {avg_at_target:.2f}dB.\n" +
            f"Frequency range with low S11 magnitude = "+ 
            (f"{frequency_range[0]:.2f} GHz to {frequency_range[1]:.2f} GHz.\n" if frequency_range is not None else "None.\n") +
            f"There exists frequencies with lower S11 magnitude values: troughs of S11 at {better_freq_list} GHz with {better_s11_values} dB."
        )

        return result_report


if __name__ == "__main__":
    import pickle
    # Load the target_spec and results from the pickle file
    with open('evaluator_test.pkl', 'rb') as f:
        target_spec, results = pickle.load(f)
    
    evaluator = SimResultsEvaluator(target_spec, results)

    report = evaluator.parse_results()

    print(report)


