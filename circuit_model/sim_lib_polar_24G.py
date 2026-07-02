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

################################
# 1) Helper: read CSV file, pick row for given freq
################################
def read_s_param_file(filename, target_freq_ghz):
    """
    Reads a CSV like 'AutoMS-profiling-SXX_YY.csv' 
    that contains real/imag columns for multiple patterns.
    
    Returns a dict:
        {
          'freqs': [...], 
          'real_cols': [...],    # 2D list (rows x patterns)
          'imag_cols': [...],    # 2D list (rows x patterns) if in the same file
          ...
        }
    or (row_real, row_imag) for the chosen frequency, depending on your format.
    """
    rows = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        all_rows = list(reader)
        
    # The first ~2 lines might be headers
    header = all_rows[0]  # column names
    # Possibly the second line is also some header. 
    # We'll skip lines until we find numeric data.
    
    data_rows = all_rows[1:]  # everything else
    freqs = []
    # We need to store columns for real and imag. 
    # But the file might have them interleaved or appended. 
    # Let's just store them as floats in order.
    
    numeric_data = []
    for row in data_rows:
        # first column is freq
        try:
            freq_val = float(row[0])
            freqs.append(freq_val)
            # the rest are the columns
            # Convert them to float if possible
            floats = []
            for x in row[1:]:
                try:
                    floats.append(float(x))
                except:
                    floats.append(np.nan)
            numeric_data.append(floats)
        except:
            # not a numeric row
            continue
    
    # Now find the row that is closest to target_freq_ghz
    # (assuming freq in the first column is in GHz)
    freqs = np.array(freqs)
    numeric_data = np.array(numeric_data, dtype=float)  # shape (Nrows, Ncols)
    idx = np.argmin(np.abs(freqs - target_freq_ghz))
    # This picks the row nearest the target freq
    row_vals = numeric_data[idx]
    
    # Each pattern might have a real part in one column, 
    # and the imaginary part in another column, etc.
    # In your example, the file actually has real columns first, then imaginary columns.
    # So we might split them in half if we know how many columns are real vs. imag.
    # Or we see from the sample that there's a chunk of real columns, 
    # then the same number of imag columns for the same patterns.
    # 
    # Let's guess half are real, half are imag:
    n_half = len(row_vals)//2
    real_part = row_vals[:n_half]
    imag_part = row_vals[n_half:]
    
    return real_part, imag_part

################################
# 2) Build S from the 6 files for each pattern
################################
def load_metasurface_patterns(target_freq_ghz):
    """
    Reads 6 CSV files, each containing S data for multiple patterns,
    and reconstructs the 2x2 block S for each pattern.
    
    We assume same number of patterns across all 6 files,
    real and imag columns line up, etc.
    
    Returns: a list of dictionaries, each with keys S11, S12, S21, S22 (2x2 np.array).
    """
    # # Filenames (example):
    # folder = "./circuit_model/profiles/"
    # fname_s11_11 = folder + "AutoMS-profiling-S11_11.csv"
    # fname_s11_22 = folder + "AutoMS-profiling-S11_22.csv"  # or S22_22 if that’s how you name it
    # fname_s21_11 = folder + "AutoMS-profiling-S21_11.csv"
    # fname_s21_12 = folder + "AutoMS-profiling-S21_12.csv"
    # # if needed:
    # fname_s11_21 = folder + "AutoMS-profiling-S11_21.csv"
    # fname_s21_22 = folder + "AutoMS-profiling-S21_22.csv"

    # # For each, read real,imag arrays at this freq
    # r_s11_11, i_s11_11 = read_s_param_file(fname_s11_11, target_freq_ghz)
    # r_s11_22, i_s11_22 = read_s_param_file(fname_s11_22, target_freq_ghz)
    # r_s11_21, i_s11_21 = read_s_param_file(fname_s11_21, target_freq_ghz)
    # r_s21_11, i_s21_11 = read_s_param_file(fname_s21_11, target_freq_ghz)
    # r_s21_12, i_s21_12 = read_s_param_file(fname_s21_12, target_freq_ghz)
    # r_s21_22, i_s21_22 = read_s_param_file(fname_s21_22, target_freq_ghz)

    # n_patterns = len(r_s11_11)  # assume they match

    S11_11, S11_22, S11_12, S11_21, S21_11, S21_22, S21_12, S21_21, \
         S22_11, S22_22, S22_12, S22_21, S12_11, S12_22, S12_12, S12_21 = get_pattern_params(target_freq_ghz)
    n_patterns = len(S11_11)  # assume they match

    patterns = []
    for idx in range(n_patterns):
        # # build each 2x2 S11, S21
        # S11_xx = (r_s11_11[idx] + 1j*i_s11_11[idx])  # S11, X->X
        # S11_yy = (r_s11_22[idx] + 1j*i_s11_22[idx])  # S11, Y->Y
        # S11_xy = (r_s11_21[idx] + 1j*i_s11_21[idx])  # S11, X->Y
        # # if cross terms are negligible or the file S11_21 means something else, 
        # # you'd parse them similarly. Let's assume we only have diagonal for S11?

        # S21_xx = (r_s21_11[idx] + 1j*i_s21_11[idx])  # S21, X->X
        # S21_xy = (r_s21_12[idx] + 1j*i_s21_12[idx])  # S21, X->Y
        # # if S21_yy is from the last file:
        # S21_yy = (r_s21_22[idx] + 1j*i_s21_22[idx])  # S21, Y->Y
        # # Y->X could come from S21_21 if needed.

        # 2x2 block for S11:
        S11_block = np.array([[S11_11[idx],  S11_12[idx]],
                              [S11_21[idx],  S11_22[idx]]], dtype=complex)
        # 2x2 block for S21:
        # we have X->X in [0,0], X->Y in [0,1], Y->X in [1,0], Y->Y in [1,1], etc.
        # Let's assume Y->X is 0 for now if not measured. 
        # Or if we have it, fill it in properly.
        S21_block = np.array([[S21_11[idx], S21_12[idx]],
                              [S21_21[idx], S21_22[idx]]], dtype=complex)
        
        S22_block = np.array([[S22_11[idx], S22_12[idx]],
                              [S22_21[idx], S22_22[idx]]], dtype=complex)
        
        S12_block = np.array([[S12_11[idx], S12_12[idx]],
                                [S12_21[idx], S12_22[idx]]], dtype=complex)

        # So we can store them in a dictionary
        pattern_dict = {
            'S11': S11_block,
            'S12': S12_block,  
            'S21': S21_block,
            'S22': S22_block
        }
        patterns.append(pattern_dict)

    return patterns


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


def find_good_variations(target_freq_ghz, target_gap = 7e-3):
    global variation_list

    # target_freq_ghz = 24.0
    c0 = 3e8
    freq_hz = target_freq_ghz*1e9
    wavelength = c0/freq_hz
    k0 = 2*np.pi / wavelength

    # 1) Load metasurface pattern data

    # load surface pattern ABCD from a pickle, if not existing, generate it
    cached_S_file = f"./circuit_model/profiles/cached_S_{target_freq_ghz:.1f}GHz.pkl"
    if not os.path.exists(cached_S_file):
        all_patterns = load_metasurface_patterns(target_freq_ghz)
        # patterns_layer2 = load_metasurface_patterns(target_freq_ghz)  # or a different file set
        with open(cached_S_file, 'wb') as f:
            pickle.dump((all_patterns, variation_list), f)
    else:
        with open(cached_S_file, 'rb') as f:
            all_patterns, variation_list = pickle.load(f)

    assert len(all_patterns) == len(variation_list), "Mismatch in pattern count and variation count"
    
    patterns_layer1 = []
    patterns_layer2 = []

    # Pattern 1 has more limited patterns
    # Special! first layer only non-rotated patterns
    patterns_layer1 = all_patterns[1:3] + all_patterns[10:12] 
    variation_list_short1 = variation_list[1:3] + variation_list[10:12]
    
    # And we remove Ring_R = 2mm to match HFSS simulation.
    variation_list_short2 = []
    for id, var in enumerate(variation_list):
        if var["Ring1_R1"] == 0.002 or var["Ring1_rotate_angle"] == 0:
            continue
        patterns_layer2.append(all_patterns[id])
        variation_list_short2.append(var)
    

    gap_list = np.linspace(0, 10e-3, 21)  # just an example of gap thicknesses

    required_cross_pol_dB = -15
    for d_gap in gap_list:
        variation_list_ret = []
        M_gap = block_air_gap(k0, d_gap)
        valid_co_pol_phases = []
        max_amp_dB = -np.inf
        for id1, pat1 in enumerate(patterns_layer1):
            M1 = s_to_abcd_2x2_blocks(pat1['S11'], pat1['S12'], pat1['S21'], pat1['S22'])
            for id2, pat2 in enumerate(patterns_layer2):
                # pat2["S11"], pat2["S12"], pat2["S21"], pat2["S22"] = np.zeros_like(pat2["S11"]), np.eye(2), np.eye(2), np.zeros_like(pat2["S22"]) # set to zero for testing
                M2 = s_to_abcd_2x2_blocks(pat2['S11'], pat2['S12'], pat2['S21'], pat2['S22'])
                # M2 = M1 # forcing M2 to be the same as M1 to test
                M_total = cascade_4x4([M1, M_gap, M2])
                # M_total = M1
                S11, S12, S21, S22 = abcd_to_s_2x2_blocks(M_total)

                # cross-pol: X->Y is S21[0,1], for instance
                cross_pol_amp_dB = db(S21[0, 1])  
                if cross_pol_amp_dB > required_cross_pol_dB:
                    if cross_pol_amp_dB > max_amp_dB:
                        max_amp_dB = cross_pol_amp_dB

                    # print out the variation index of the pattern
                    # print(f"Pattern 1: {id1}")
                    # print(f"Pattern 2: {id2}")
                    # print(f"Corresponding variation: {variation_list_short[id1]}, {variation_list[id2]}")
                    variation_list_ret.append((variation_list_short1[id1], variation_list_short2[id2]))
                    co_pol_phase = np.angle(S21[0, 1]) # cross-pol phase
                    valid_co_pol_phases.append(co_pol_phase)
        
        print(f"Gap={d_gap*1000:.1f} mm, max amplitude = {max_amp_dB} dB")

        if len(valid_co_pol_phases) < 2:
            phase_range = 0
        else:
            phase_range = np.max(valid_co_pol_phases) - np.min(valid_co_pol_phases)
            print(f"Gap={d_gap*1000:.1f} mm, #valid combos={len(valid_co_pol_phases)}, "
              f"co-pol phase range={np.degrees(phase_range):.1f} deg")
        
        if d_gap == target_gap:
            return variation_list_ret

###############################
# MAIN DEMO
###############################
if __name__ == "__main__":
    global variation_list
    variation_list =[]

    # find the overlap of good variations for 24 and 25 GHz
    allGood_variation_list = []
    for target_freq in [23, 24, 25]:
        print(f"\n+++++++ Target freq = {target_freq} GHz +++++++")
        cur_var = find_good_variations(target_freq, target_gap=10e-3)
        if allGood_variation_list == []:
            allGood_variation_list = cur_var
        else:
            for var in allGood_variation_list:
                if var not in cur_var:
                    allGood_variation_list.remove(var)

    print("\n+++++++ Good variations +++++++")
    # print(allGood_variation_list)
    print(len(allGood_variation_list))
   
