import numpy as np



def z_to_abcd_2x2_blocks(Z11, Z12, Z21, Z22):
    """
    Convert the 2x2-block Z-parameters (each block is 2x2) into a 4x4 ABCD matrix.
    
    Arguments:
      Z11, Z12, Z21, Z22 : 2x2 numpy arrays (complex)
         Blocks of the overall 4x4 Z-matrix, i.e.:
            [ [Z11, Z12],
              [Z21, Z22] ]
         where each Z_ij is 2x2.
    
    Returns:
      M_4x4 : 4x4 numpy array (complex)
         The ABCD (transfer) matrix in block form:
            [ [ A, B ],
              [ C, D ] ]
         where each of A, B, C, D is 2x2.
    """
    # Invert Z21 (2x2)
    Z21_inv = np.linalg.inv(Z21)

    # Compute each 2x2 block in the ABCD matrix
    A_block = Z11 @ Z21_inv
    B_block = Z21_inv @ (Z11 @ Z22 - Z12 @ Z21)
    C_block = Z21_inv
    D_block = Z21_inv @ Z22

    # Assemble into a 4x4 by stacking 2x2 blocks
    top_row = np.hstack((A_block, B_block))  # shape (2,4)
    bot_row = np.hstack((C_block, D_block))  # shape (2,4)
    M_4x4 = np.vstack((top_row, bot_row))     # shape (4,4)

    return M_4x4

# # --------------------- Example usage --------------------- #
# if __name__ == "__main__":
#     # Dummy example: each Z_ij is 2x2
#     Z11 = np.array([[1+1j,   0   ],
#                     [0,      1.5j]], dtype=complex)
#     Z12 = np.array([[0,   0],
#                     [0,   0]], dtype=complex)  # no cross coupling for example
#     Z21 = np.array([[0.5j,  0   ],
#                     [0,     0.3j]], dtype=complex)
#     Z22 = np.array([[2,    0 ],
#                     [0,    3 ]], dtype=complex)

#     M_4x4 = z_to_abcd_2x2_blocks(Z11, Z12, Z21, Z22)

#     print("ABCD matrix (4x4) =\n", M_4x4)


def s_to_abcd_2x2_blocks(S11, S12, S21, S22):
    """
    Convert 2x2-block S-parameters (each block is 2x2) into a 4x4 ABCD matrix.
    
    Parameters:
      S11, S12, S21, S22 : 2x2 numpy arrays
         Represent the measured reflection/transmission in block form.
         For example, S11 is the 2x2 reflection block, etc.
    
    Returns:
      M : 4x4 numpy array
         The block-ABCD (a transfer matrix form).
    """

    # Invert the S21 block (2x2)
    S21_inv = np.linalg.inv(S21)

    # Build each 2x2 sub-block of the ABCD matrix
    # One common formula is:
    # A = - S21^-1 * S22
    # B = - S21^-1
    # C = I - S11 * S21^-1 * S22
    # D = - S11 * S21^-1

    A_block = - S21_inv @ S22
    B_block = - S21_inv
    C_block = np.eye(2, dtype=complex) - S11 @ S21_inv @ S22
    D_block = - S11 @ S21_inv

    # Now assemble them into a 4x4
    # We'll do the block structure:
    #
    # [ A_block   B_block ]
    # [ C_block   D_block ]

    M_top = np.hstack((A_block, B_block))   # 2x4
    M_bot = np.hstack((C_block, D_block))   # 2x4
    M = np.vstack((M_top, M_bot))           # 4x4

    return M


# Example usage:
if __name__ == "__main__":
    # Suppose each S_ij is a 2x2 array you measured, e.g. cross/ co-polar terms
    # Quick dummy data for demonstration:
    S11 = np.array([[0.1+0.2j, 0.0      ],
                    [0.0,      0.05+0.1j]])
    S21 = np.array([[0.92, 0.0 ],
                    [0.0,  0.85]], dtype=complex)
    
    # as we are reciprocal, S12 = S21
    S12 = S21
    S22 = S11

    M_4x4 = s_to_abcd_2x2_blocks(S11, S12, S21, S22)
    print("4x4 ABCD matrix =\n", M_4x4)

