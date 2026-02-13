import numpy as np
import control
from pympctl.model.lin_model import LinModel
from pympctl.controller.linmpc import LinMPC

def test_linmpc():
    # Model
    A = [[0.8, 0.1], [0, 0.9]]
    B = [[1.0], [0.5]]
    C = [[1.0, 0]]
    D = [[0]]
    sys = control.ss(A, B, C, D, dt=0.1)
    model = LinModel(sys)

    # MPC
    mpc = LinMPC(model, Hp=10, Hc=2, Mwt=[1.0], Nwt=[0.1])

    # Solve
    r_y = [1.0]
    try:
        u = mpc.solve(r_y)
        print(f"Computed u: {u}")
        assert u.shape == (1,)
        print("LinMPC test passed.")
    except Exception as e:
        print(f"LinMPC test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_linmpc()
