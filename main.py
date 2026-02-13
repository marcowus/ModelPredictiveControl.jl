import numpy as np
import control
import matplotlib.pyplot as plt
from pympctl.model.lin_model import LinModel
from pympctl.controller.linmpc import LinMPC
from pympctl.sim.simulation import sim

def main():
    Ts = 1.0

    # Define systems
    sys1_c = control.tf([2], [10, 1])
    sys2_c = control.tf([10], [4, 1])

    # Discretize
    sys1_d = control.c2d(sys1_c, Ts, method='zoh')
    sys2_d = control.c2d(sys2_c, Ts, method='zoh')

    # Add delay to sys1
    # z^-20 = 1 / z^20
    # Coefficients: [1] for num, [1, 0...0] for den.
    delay_steps = 20
    num_delay = [1]
    den_delay = [1] + [0]*delay_steps
    delay_tf = control.tf(num_delay, den_delay, dt=Ts)

    sys1_delayed = sys1_d * delay_tf

    # Convert to State Space individually
    sys1_ss = control.ss(sys1_delayed)
    sys2_ss = control.ss(sys2_d)

    # Combine into MIMO manually (1 input, 2 outputs)
    # x = [x1; x2]
    # u = u
    # y = [y1; y2]
    # A = [A1 0; 0 A2]
    # B = [B1; B2]
    # C = [C1 0; 0 C2]
    # D = [D1; D2]

    nx1 = sys1_ss.nstates
    nx2 = sys2_ss.nstates

    A = np.block([[sys1_ss.A, np.zeros((nx1, nx2))],
                  [np.zeros((nx2, nx1)), sys2_ss.A]])

    B = np.vstack([sys1_ss.B, sys2_ss.B])

    C = np.block([[sys1_ss.C, np.zeros((1, nx2))],
                  [np.zeros((1, nx1)), sys2_ss.C]])

    D = np.vstack([sys1_ss.D, sys2_ss.D])

    sys_ss = control.ss(A, B, C, D, dt=Ts)

    # Minimal realization to remove uncontrollable/unobservable states if any
    # But separate construction ensures minimal if components are minimal.
    # Delay states are controllable/observable.

    print(f"System states: {sys_ss.nstates}")

    # Initialize LinModel
    model = LinModel(sys_ss, Ts)

    # Initialize MPC
    # Hp=40 to cover delay.
    mpc = LinMPC(model, Hp=40, Hc=2, Mwt=[1.0, 0.0], Nwt=[0.1])
    mpc.setconstraint(ymax=[np.inf, 35])

    # Simulation
    # Setpoint ry = [5, 0]. y1 -> 5.
    ry = [5, 0]
    N = 80

    print("Starting simulation...")
    res = sim(mpc, N, ry)
    print("Simulation finished.")

    # Plot
    t = res['T']
    y = res['Y']
    u = res['U']
    r = res['R']

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.step(t, y[:, 0], where='post', label='y1')
    plt.step(t, y[:, 1], where='post', label='y2')
    # plt.step(t, r[:, 0], where='post', linestyle='--', label='r1')
    plt.axhline(5, color='k', linestyle='--', label='r1')
    plt.axhline(35, color='r', linestyle='--', label='y2 max')
    plt.ylabel('Outputs')
    plt.legend()
    plt.title('MPC Simulation Results')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.step(t, u[:, 0], where='post', label='u')
    plt.ylabel('Input')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('mpc_result.png')
    print("Result saved to mpc_result.png")

    # Check results
    final_y1 = y[-1, 0]
    final_y2 = y[-1, 1]
    max_y2 = np.max(y[:, 1])

    print(f"Final y1: {final_y1:.4f} (Target: 5.0)")
    print(f"Final y2: {final_y2:.4f}")
    print(f"Max y2: {max_y2:.4f} (Constraint: 35.0)")

    if np.abs(final_y1 - 5.0) < 0.1:
        print("y1 tracking successful.")
    else:
        print("y1 tracking failed.")

    if max_y2 <= 35.1: # Allow small tolerance for soft constraints or solver precision
        print("y2 constraint satisfied.")
    else:
        print("y2 constraint violated.")

if __name__ == "__main__":
    main()
