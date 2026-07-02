import numpy as np

import numpy as np

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
    
    return A, B, C, D

def abcd_to_s_2x2_blocks(A, B, C, D, Z0=377.0):
    """
    Convert ABCD parameters (each a 2x2 matrix) back to S-parameters.
    """
    I = np.eye(2, dtype=complex)
    # denom is a 2x2 matrix
    denom = A + (B / Z0) + (Z0 * C) + D
    denom_inv = np.linalg.inv(denom)
    
    S11 = (A + B / Z0 - Z0 * C - D) @ denom_inv
    S21 = 2 * denom_inv
    S12 = 2 * (A @ D - B @ C) @ denom_inv
    S22 = (-A + B / Z0 - Z0 * C + D) @ denom_inv
    
    return S11, S12, S21, S22

# Example test (using dummy S-parameter matrices):
if __name__ == "__main__":
    I = np.eye(2, dtype=complex)
    # Create some example S-parameters (they must be physically valid in a real case)
    S11 = 0.1 * I
    S22 = 0.1 * I
    S12 = 0.2 * I
    S21 = 0.8 * I

    A, B, C, D = s_to_abcd_2x2_blocks(S11, S12, S21, S22)
    S11_rec, S12_rec, S21_rec, S22_rec = abcd_to_s_2x2_blocks(A, B, C, D)

    print("Original S11:\n", S11)
    print("Recovered S11:\n", S11_rec)
    print("Original S21:\n", S21)
    print("Recovered S21:\n", S21_rec)


    import numpy as np
    from numpy.random import rand
    np.set_printoptions(precision=3, suppress=True)

    # Make up random invertible S21
    while True:
        S21 = (rand(2,2) + 1j*rand(2,2))
        if abs(np.linalg.det(S21)) > 0.01:
            break

    # Make the rest random
    S11 = (rand(2,2)+1j*rand(2,2))
    S12 = (rand(2,2)+1j*rand(2,2))
    S22 = (rand(2,2)+1j*rand(2,2))

    # Convert S->T->S'
    A, B, C, D = s_to_abcd_2x2_blocks(S11, S12, S21, S22)
    S11p, S12p, S21p, S22p = abcd_to_s_2x2_blocks(A, B, C, D)

    print("Original S21:\n", S21)
    print("Recovered S21:\n", S21p)
    print("Diff norm = ", np.linalg.norm(S21-S21p))
    print("Same? ->", np.allclose(S21, S21p, atol=1e-8))

    # likewise check S11, S12, S22
    print("Diff norm S11 =", np.linalg.norm(S11-S11p))
    print("Diff norm S12 =", np.linalg.norm(S12-S12p))
    print("Diff norm S22 =", np.linalg.norm(S22-S22p))
