import numpy as np
import control
from pympctl.model.lin_model import LinModel

def test_linmodel():
    # Define a simple continuous system
    A = [[-0.1, 1.0], [1.0, 0]]
    B = [[1.0], [0]]
    C = [[1.0, 0]]
    D = [[0]]
    sys = control.ss(A, B, C, D)
    Ts = 0.1

    # Initialize LinModel
    model = LinModel(sys, Ts)

    print("LinModel initialized successfully.")
    print(f"A shape: {model.A.shape}")
    print(f"Bu shape: {model.Bu.shape}")
    print(f"C shape: {model.C.shape}")

    # Check dimensions
    assert model.nx == 2
    assert model.nu == 1
    assert model.ny == 1

    # Check discretization happened (A should not be same as continuous A if Ts is small)
    assert not np.allclose(model.A, A)

    print("LinModel test passed.")

if __name__ == "__main__":
    test_linmodel()
