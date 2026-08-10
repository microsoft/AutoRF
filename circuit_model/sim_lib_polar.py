import numpy as np
import matplotlib.pyplot as plt

# ---- 2) Helper: 4x4 transmission line for air gap ---- #
def block_air_gap(k, d, Z0=377.0):
    """
    Builds a 4x4 block for an air gap (thickness d, wave number k0).
    Assumes same wave impedance in x,y => 2x2 identity * Z0
    """
    I = np.eye(2, dtype=complex)
    Z = np.diag([Z0, Z0])

    cos_kd = np.cos(k*d)
    sin_kd = np.sin(k*d)

    A = cos_kd * I
    B = 1j * sin_kd * Z
    C = 1j * sin_kd * np.linalg.inv(Z)
    D = cos_kd * I

    return np.block([
        [A, B],
        [C, D]
    ])


def db(mag):
    """Convert linear magnitude to dB."""
    return 20*np.log10(np.abs(mag) + 1e-30)



import csv
import numpy as np

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.hfss_manager import *


def get_s_param_data(hfss, cur_expr: str, target_freq:float, primary_var: str = "Freq", variations: list = {"Freq": ["All"], "Ring1_R1": ["All"], "Ring1_rotate_angle": ["All"], "Ring1_Rw": ["All"]}):
# self.logger.debug(f"Getting HFSS results with expression: {cur_expr}")

    result_obj = hfss.post.get_solution_data(
        expressions=cur_expr,
        primary_sweep_variable=primary_var,
        variations=variations,
        report_category="Modal Solution Data",
    )
    global variation_list
    if variation_list == []:  # first time
        variation_list = result_obj.variations
    else:
        assert variation_list == result_obj.variations, "Variations don't match"


    # Find the index of the target frequency
    freqs = np.array(result_obj.primary_sweep_values)
    idx = np.argmin(np.abs(freqs - target_freq))
    
    data_lines = []
    for id, var in enumerate(result_obj.variations):
        # self.logger.debug(f"Variation {id}: {var}")
            
        
        result_obj.set_active_variation(id)
        assert result_obj.active_variation == var, f"Failed to set active variation to {id}"
        data_lines.append(np.array(result_obj.data_real()) + 1j* np.array(result_obj.data_imag()))

    # if cur_expr == "S(FloquetPort2:1,FloquetPort1:2)":
    #     # plot the S21 amplitude dB
    #     S21 = np.array(data_lines)
    #     S21_amp = db(np.abs(S21))
    #     for id, var in enumerate(result_obj.variations):
    #         plt.plot(freqs, S21_amp[id, :], label=f"Var {var}")
    #     plt.xlabel("Frequency (GHz)")
    #     plt.ylabel("S21 amplitude (dB)")
    #     plt.title("S21 amplitude vs. frequency")
    #     plt.show()

    # Get the values of all variations at this frequency
    data_lines = np.array(data_lines)
    values = data_lines[:, idx]


    return values

def get_pattern_params(target_freq_ghz):
    hfss = ansys.aedt.core.Hfss(
        version="2024.2",
        project=None, # Find a default active project
        design="24G-trans-rings-new-1layer-profiling",  # find a default design in the project
        # solution_type=self.default_settings["solver_type"],
        solution_type=None,
    )

    
    ports = hfss.ports  # Get all port names
    
    S11_11 = get_s_param_data(hfss, f"S({ports[0]}:1,{ports[0]}:1)", target_freq_ghz)
    S11_22 = get_s_param_data(hfss, f"S({ports[0]}:2,{ports[0]}:2)", target_freq_ghz)
    S11_12 = get_s_param_data(hfss, f"S({ports[0]}:1,{ports[0]}:2)", target_freq_ghz)
    S11_21 = get_s_param_data(hfss, f"S({ports[0]}:2,{ports[0]}:1)", target_freq_ghz)

    S21_11 = get_s_param_data(hfss, f"S({ports[1]}:1,{ports[0]}:1)", target_freq_ghz)
    S21_22 = get_s_param_data(hfss, f"S({ports[1]}:2,{ports[0]}:2)", target_freq_ghz)
    S21_12 = get_s_param_data(hfss, f"S({ports[1]}:1,{ports[0]}:2)", target_freq_ghz)
    S21_21 = get_s_param_data(hfss, f"S({ports[1]}:2,{ports[0]}:1)", target_freq_ghz)

    S22_11 = get_s_param_data(hfss, f"S({ports[1]}:1,{ports[1]}:1)", target_freq_ghz)
    S22_22 = get_s_param_data(hfss, f"S({ports[1]}:2,{ports[1]}:2)", target_freq_ghz)
    S22_12 = get_s_param_data(hfss, f"S({ports[1]}:1,{ports[1]}:2)", target_freq_ghz)
    S22_21 = get_s_param_data(hfss, f"S({ports[1]}:2,{ports[1]}:1)", target_freq_ghz)

    S12_11 = get_s_param_data(hfss, f"S({ports[0]}:1,{ports[1]}:1)", target_freq_ghz)
    S12_22 = get_s_param_data(hfss, f"S({ports[0]}:2,{ports[1]}:2)", target_freq_ghz)
    S12_12 = get_s_param_data(hfss, f"S({ports[0]}:1,{ports[1]}:2)", target_freq_ghz)
    S12_21 = get_s_param_data(hfss, f"S({ports[0]}:2,{ports[1]}:1)", target_freq_ghz)


    hfss.release_desktop(close_projects=False, close_desktop=False)


    return S11_11, S11_22, S11_12, S11_21, S21_11, S21_22, S21_12, S21_21, \
              S22_11, S22_22, S22_12, S22_21, S12_11, S12_22, S12_12, S12_21




def cascade_4x4(matrices):
    M = np.eye(4, dtype=complex)
    for mat in matrices:
        M = M @ mat
    return M

def s_to_abcd_2x2_blocks(S11, S12, S21, S22, Z0=377.0):
    """
    Convert S-parameters (each a 2x2 matrix) to ABCD parameters.
    """
    I = np.eye(2, dtype=complex)
    # Compute denominator: 2*S21 (matrix multiplication, note 2 is scalar)
    denom = 2 * S21
    denom_inv = np.linalg.inv(denom)
    
    A = ((I + S11) @ (I - S22) + S12 @ S21) @ denom_inv
    B = Z0 * (((I + S11) @ (I + S22) - S12 @ S21)) @ denom_inv
    C = (1 / Z0) * (((I - S11) @ (I - S22) - S12 @ S21)) @ denom_inv
    D = ((I - S11) @ (I + S22) + S12 @ S21) @ denom_inv

    # stack them into a 4x4 matrix
    M = np.block([
        [A, B],
        [C, D]
    ])
    
    return M

def abcd_to_s_2x2_blocks(M, Z0=377.0):
    """
    Convert ABCD parameters (each a 2x2 matrix) back to S-parameters.
    """
    A = M[0:2, 0:2]
    B = M[0:2, 2:4]
    C = M[2:4, 0:2]
    D = M[2:4, 2:4]

    I = np.eye(2, dtype=complex)
    # denom is a 2x2 matrix
    denom = A + (B / Z0) + (Z0 * C) + D
    denom_inv = np.linalg.inv(denom)
    
    S11 = (A + B / Z0 - Z0 * C - D) @ denom_inv
    S21 = 2 * denom_inv
    S12 = 2 * (A @ D - B @ C) @ denom_inv
    S22 = (-A + B / Z0 - Z0 * C + D) @ denom_inv
    
    return S11, S12, S21, S22


def db(x):
    return 20*np.log10(np.abs(x) + 1e-30)



import pickle
# if __name__ == "__main__":
#     get_pattern_params(24.0)


def find_good_variations(target_freq_ghz):
    print(f"+++++++ Target freq = {target_freq_ghz} GHz +++++++")

    global variation_list

    # target_freq_ghz = 24.0
    c0 = 3e8
    freq_hz = target_freq_ghz*1e9
    wavelength = c0/freq_hz
    k0 = 2*np.pi / wavelength


    gap_list = np.linspace(0, 10e-3, 21)  # just an example of gap thicknesses

    required_cross_pol_dB = -4 # dB, minimal cross-pol amplitude required.
    required_min_combo = 10 # minimal number of valid combinations for a gap

    final_best_gap = None
    best_mag_dB = -np.inf
    best_phase_range = 0


    # 1) Load metasurface pattern data

    available_target_freq_ghz = 24.0 # GHz, the one we have data for, should not change.
    # load surface pattern ABCD from a pickle, if not existing, generate it
    cached_S_file = f"./circuit_model/profiles/cached_S_{available_target_freq_ghz:.1f}GHz.pkl"
    assert os.path.exists(cached_S_file), f"Cached S file {cached_S_file} not found."

    with open(cached_S_file, 'rb') as f:
        all_patterns, variation_list = pickle.load(f)

    assert len(all_patterns) == len(variation_list), "Mismatch in pattern count and variation count"
    
    patterns_layer1 = all_patterns
    patterns_layer2 = all_patterns

    # # Pattern 1 has more limited patterns
    # # Special! first layer only non-rotated patterns
    # patterns_layer1 = all_patterns[1:3] + all_patterns[10:12] 
    # variation_list_short1 = variation_list[1:3] + variation_list[10:12]
    
    # # And we remove Ring_R = 2mm to match HFSS simulation.
    # variation_list_short2 = []
    # for id, var in enumerate(variation_list):
    #     if var["Ring1_R1"] == 0.002 or var["Ring1_rotate_angle"] == 0:
    #         continue
    #     patterns_layer2.append(all_patterns[id])
    #     variation_list_short2.append(var)
    


    for d_gap in gap_list:
        M_gap = block_air_gap(k0, d_gap)
        valid_co_pol_phases = []
        max_amp_dB = -np.inf
        for id1, pat1 in enumerate(patterns_layer1):
            M1 = s_to_abcd_2x2_blocks(pat1['S11'], pat1['S12'], pat1['S21'], pat1['S22'])
            for id2, pat2 in enumerate(patterns_layer2):
                M2 = s_to_abcd_2x2_blocks(pat2['S11'], pat2['S12'], pat2['S21'], pat2['S22'])
                M_total = cascade_4x4([M1, M_gap, M2])
                # M_total = M1
                S11, S12, S21, S22 = abcd_to_s_2x2_blocks(M_total)

                # cross-pol: X->Y is S21[0,1], for instance
                cross_pol_amp_dB = db(S21[0, 1]) 
                # ONlY consider the variations with high cross-pol amplitude, lager than the required threshold
                if cross_pol_amp_dB > required_cross_pol_dB:
                    if cross_pol_amp_dB > max_amp_dB:
                        max_amp_dB = cross_pol_amp_dB

                    # print out the variation index of the pattern
                    # print(f"Pattern 1: {id1}")
                    # print(f"Pattern 2: {id2}")
                    # print(f"Corresponding variation: {variation_list_short[id1]}, {variation_list[id2]}")
                    co_pol_phase = np.angle(S21[0, 1]) # cross-pol phase
                    valid_co_pol_phases.append(co_pol_phase)
        
        print(f"Gap={d_gap*1000:.1f} mm, max amplitude = {max_amp_dB} dB") 

        if len(valid_co_pol_phases) < 2:
            phase_range = 0
        else:
            phase_range = np.degrees(np.max(valid_co_pol_phases) - np.min(valid_co_pol_phases))
            
            print(f"Gap={d_gap*1000:.1f} mm, #valid combos={len(valid_co_pol_phases)}, "
              f"co-pol phase range={phase_range:.1f} deg")
            
        if len(valid_co_pol_phases) > required_min_combo and phase_range > 300:
            # Check if this gap meets a phase range > 270 deg and sufficient valid combos.
            if max_amp_dB > best_mag_dB: # max amplitude check, want to maximize the amplitude as multi layer surface can have high loss issue.
                best_mag_dB = max_amp_dB
                best_phase_range = phase_range
                final_best_gap = d_gap
        
    
    print(f"Best gap={final_best_gap*1000:.1f} mm, phase range={best_phase_range:.1f} deg")


###############################
# MAIN DEMO
###############################
if __name__ == "__main__":
    global variation_list
    variation_list =[]

    target_freq_ghz = 79
    find_good_variations(target_freq_ghz)

    
