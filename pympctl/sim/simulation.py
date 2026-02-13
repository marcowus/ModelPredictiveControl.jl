import numpy as np

def sim(mpc, N, r_y, d=None):
    """
    Simulate the closed-loop system.

    Args:
        mpc: LinMPC object.
        N: Number of simulation steps.
        r_y: Setpoint (can be time-varying array or scalar).
        d: Disturbance (can be time-varying array or scalar).

    Returns:
        res: Dictionary with simulation results (T, U, Y, X, R).
    """
    model = mpc.model
    Ts = model.Ts

    ny = model.ny
    nu = model.nu
    nx = model.nx

    # Initialize storage
    U_data = np.zeros((N, nu))
    Y_data = np.zeros((N, ny))
    X_data = np.zeros((N, nx))
    R_data = np.zeros((N, ny))
    T_data = np.arange(N) * Ts

    # Process setpoint
    if np.isscalar(r_y):
        r_y = np.full((N, ny), r_y)
    else:
        r_y = np.array(r_y)
        if r_y.ndim == 1:
            if ny == 1:
                 r_y = r_y.reshape(-1, 1)
                 if r_y.shape[0] < N:
                      r_y = np.vstack([r_y, np.tile(r_y[-1], (N-r_y.shape[0], 1))])
            else:
                 if r_y.size == ny:
                     # Assume constant vector
                     r_y = np.tile(r_y, (N, 1))
                 else:
                     raise ValueError(f"r_y dimension mismatch. Expected {ny} elements or time series.")
        elif r_y.ndim == 2:
             if r_y.shape[1] != ny:
                  raise ValueError(f"r_y width {r_y.shape[1]} does not match ny {ny}")
             if r_y.shape[0] < N:
                  r_y = np.vstack([r_y, np.tile(r_y[-1], (N-r_y.shape[0], 1))])

    # Initial state
    x = model.x0.copy()

    # Loop
    for k in range(N):
        # Current output
        y = model.C @ x
        # Add disturbance effect if implemented

        # Store data
        Y_data[k] = y
        X_data[k] = x
        R_data[k] = r_y[k]

        # MPC Step
        # Update model state in MPC (assuming perfect state feedback)
        mpc.model.x0 = x

        # Construct reference vector for MPC
        Hp = mpc.Hp
        if k + Hp <= N:
             ref = r_y[k : k + Hp].flatten()
        else:
             ref = np.vstack([r_y[k:], np.tile(r_y[-1], (Hp - (N-k), 1))]).flatten()

        # Solve
        u = mpc.solve(ref)

        if u is None:
             print(f"MPC failed at step {k}")
             if k > 0:
                 u = U_data[k-1]
             else:
                 u = np.zeros(nu)

        U_data[k] = u

        # Simulate Plant
        # x_next = A x + B u
        x_next = model.A @ x + model.Bu @ u
        x = x_next

    return {
        "T": T_data,
        "U": U_data,
        "Y": Y_data,
        "X": X_data,
        "R": R_data
    }
