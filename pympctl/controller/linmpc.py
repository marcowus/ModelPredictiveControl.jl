import numpy as np
import cvxpy as cp
from pympctl.model.lin_model import LinModel

class LinMPC:
    def __init__(self, model: LinModel, Mwt=None, Nwt=None, Lwt=None, Hp=10, Hc=2,
                 ymin=None, ymax=None, umin=None, umax=None,
                 dumin=None, dumax=None):
        """
        Linear MPC Controller.

        Args:
            model: LinModel object.
            Mwt: Output tracking weights (vector of length ny).
            Nwt: Input increment weights (vector of length nu).
            Lwt: Input tracking weights (vector of length nu).
            Hp: Prediction horizon.
            Hc: Control horizon.
            ymin, ymax: Output bounds.
            umin, umax: Input bounds.
            dumin, dumax: Input increment bounds.
        """
        self.model = model
        self.Hp = Hp
        self.Hc = Hc

        # Default weights
        if Mwt is None: Mwt = np.ones(model.ny)
        if Nwt is None: Nwt = np.array([0.1]*model.nu)
        if Lwt is None: Lwt = np.zeros(model.nu)

        self.Mwt = np.atleast_1d(Mwt)
        self.Nwt = np.atleast_1d(Nwt)
        self.Lwt = np.atleast_1d(Lwt)

        # Constraints
        # Constraints are stored as full vectors over horizon? Or just single values repeated?
        # The Julia code supports time-varying constraints but default is constant.
        # We will store the bounds provided, assuming they are constant over horizon or vector.
        # But for now, let's just store them as is and handle repetition in solve/matrix generation.

        self.ymin = np.atleast_1d(ymin) if ymin is not None else np.full(model.ny, -np.inf)
        self.ymax = np.atleast_1d(ymax) if ymax is not None else np.full(model.ny, np.inf)
        self.umin = np.atleast_1d(umin) if umin is not None else np.full(model.nu, -np.inf)
        self.umax = np.atleast_1d(umax) if umax is not None else np.full(model.nu, np.inf)
        self.dumin = np.atleast_1d(dumin) if dumin is not None else np.full(model.nu, -np.inf)
        self.dumax = np.atleast_1d(dumax) if dumax is not None else np.full(model.nu, np.inf)

        # Initialize internal variables
        self.last_u = np.zeros(model.nu)

        # Placeholder for matrices
        self.E = None
        self.G = None
        self.J = None
        self.K = None
        self.V = None
        self.B = None

        # Call init_predmat to setup matrices
        self.init_predmat()

    def setconstraint(self, ymin=None, ymax=None, umin=None, umax=None, dumin=None, dumax=None):
        if ymin is not None: self.ymin = np.atleast_1d(ymin)
        if ymax is not None: self.ymax = np.atleast_1d(ymax)
        if umin is not None: self.umin = np.atleast_1d(umin)
        if umax is not None: self.umax = np.atleast_1d(umax)
        if dumin is not None: self.dumin = np.atleast_1d(dumin)
        if dumax is not None: self.dumax = np.atleast_1d(dumax)
        return self

    def init_predmat(self):
        """
        Initialize prediction matrices E, F_matrix for Y = E * DeltaU + F_matrix * x_aug
        """
        A = self.model.A
        B = self.model.Bu
        C = self.model.C
        nx = self.model.nx
        nu = self.model.nu
        ny = self.model.ny
        Hp = self.Hp
        Hc = self.Hc

        # Augmented state: [x_k; u_{k-1}]
        # x_{k+1} = A x_k + B u_k
        # u_k = u_{k-1} + du_k
        # -> x_{k+1} = A x_k + B u_{k-1} + B du_k
        # -> u_k     =     0 x_k + I u_{k-1} + I du_k

        # Augmented matrices
        self.A_aug = np.block([[A, B], [np.zeros((nu, nx)), np.eye(nu)]])
        self.B_aug = np.vstack([B, np.eye(nu)])
        self.C_aug = np.hstack([C, np.zeros((ny, nu))])

        self.n_aug = nx + nu

        # F matrix (Free response) maps initial augmented state to outputs Y over Hp
        # Y = [y_{k+1}; ...; y_{k+Hp}]
        # y_{k+i} = C_aug * A_aug^i * x_aug_k

        self.F_matrix = np.zeros((ny * Hp, self.n_aug))
        A_pow = np.eye(self.n_aug)
        for i in range(1, Hp + 1):
            A_pow = A_pow @ self.A_aug
            self.F_matrix[(i-1)*ny : i*ny, :] = self.C_aug @ A_pow

        # E matrix (Forced response) maps Delta U over Hc to outputs Y over Hp
        # Delta U = [du_k; ...; du_{k+Hc-1}]
        # y_{k+i} = sum_{j=0}^{i-1} C_aug * A_aug^(i-1-j) * B_aug * du_{k+j}

        self.E = np.zeros((ny * Hp, nu * Hc))

        # Compute impulse responses (Markov parameters)
        # H_i = C_aug * A_aug^(i-1) * B_aug
        H = []
        A_pow = np.eye(self.n_aug)
        for i in range(Hp):
            # i=0 -> y_{k+1} term for du_k -> C B_aug
            H.append(self.C_aug @ A_pow @ self.B_aug)
            A_pow = A_pow @ self.A_aug

        # Construct E (Toeplitz)
        for i in range(Hp): # Row block i (y_{k+1+i})
            for j in range(Hc): # Col block j (du_{k+j})
                if j <= i:
                    self.E[i*ny : (i+1)*ny, j*nu : (j+1)*nu] = H[i-j]

        # Construct Tu matrix (maps DeltaU to U)
        # U = [u_k; ...; u_{k+Hp-1}] (Wait, usually Hc for U in optimization)
        # But we need U over Hc for constraints.
        # u_{k+i} = u_{k-1} + sum_{j=0}^i du_{k+j}
        # U = T * DeltaU + U_init

        # We need U over Hc (optimization variables) or Hp (constraints)?
        # Constraints are usually over Hp.
        # But prediction variables DeltaU are only over Hc.
        # For j >= Hc, du_{k+j} = 0, so u stays constant.

        self.Tu = np.zeros((nu * Hp, nu * Hc))
        for i in range(Hp):
            for j in range(Hc):
                if j <= i:
                    self.Tu[i*nu : (i+1)*nu, j*nu : (j+1)*nu] = np.eye(nu)

    def solve(self, r_y, d=None):
        """
        Solve the MPC optimization problem.
        r_y: Setpoint (can be scalar or vector, if scalar replicated).
        d: Measured disturbance (not implemented yet, assumed 0).
        """
        # Ensure r_y is vector over Hp or just constant
        ny = self.model.ny
        nu = self.model.nu

        # Prepare reference vector R
        if np.isscalar(r_y):
            r_y = np.full(ny, r_y)
        r_y = np.array(r_y)
        if r_y.ndim == 1:
            if r_y.size == ny:
                # Constant setpoint
                R = np.tile(r_y, self.Hp)
            elif r_y.size == ny * self.Hp:
                R = r_y
            else:
                 raise ValueError(f"r_y size mismatch. Expected {ny} or {ny*self.Hp}.")
        else:
             raise ValueError("r_y must be 1D array")

        # Current augmented state
        # x_aug_k = [x_k; u_{k-1}]
        x_k = self.model.x0
        u_km1 = self.last_u

        # Get operating points
        uop = self.model.uop
        yop = self.model.yop
        xop = self.model.xop

        # x_k in model is deviation (LinModel logic).
        # u_km1 stored in controller is absolute.
        # Deviation u_km1_dev
        u_km1_dev = u_km1 - uop
        x_k_dev = x_k # model.x0 is already deviation

        x_aug_k = np.concatenate([x_k_dev, u_km1_dev])

        # Free response
        Y_free = self.F_matrix @ x_aug_k

        # Optimization variables: Delta U
        dU = cp.Variable(nu * self.Hc)

        # Predicted Output
        Y_pred = self.E @ dU + Y_free

        # Reference R is absolute. Convert to deviation.
        R_dev = R - np.tile(yop, self.Hp)

        # Cost matrices
        M_diag = np.tile(self.Mwt, self.Hp)
        N_diag = np.tile(self.Nwt, self.Hc)

        # Cost
        # Error: Y_pred - R_dev
        error = Y_pred - R_dev

        cost = cp.quad_form(error, np.diag(M_diag)) + cp.quad_form(dU, np.diag(N_diag))

        # Constraints
        constraints = []

        # Input constraints
        # u_{min} <= u_{op} + u_{dev} <= u_{max}
        # u_{dev} = Tu * dU + U_init_dev

        U_init_dev = np.tile(u_km1_dev, self.Hp)
        U_dev = self.Tu @ dU + U_init_dev

        # u_{min} - u_{op} <= U_dev <= u_{max} - u_{op}

        U_min_vec = np.tile(self.umin, self.Hp)
        U_max_vec = np.tile(self.umax, self.Hp)

        U_op_vec = np.tile(uop, self.Hp)

        constraints += [U_dev >= U_min_vec - U_op_vec]
        constraints += [U_dev <= U_max_vec - U_op_vec]

        # Input increment constraints
        # dU corresponds to Delta u.
        # dU_{min} <= dU <= dU_{max}

        dU_min_vec = np.tile(self.dumin, self.Hc)
        dU_max_vec = np.tile(self.dumax, self.Hc)

        constraints += [dU >= dU_min_vec]
        constraints += [dU <= dU_max_vec]

        # Output constraints
        # y_{min} <= y_{op} + Y_pred <= y_{max}

        Y_min_vec = np.tile(self.ymin, self.Hp)
        Y_max_vec = np.tile(self.ymax, self.Hp)

        Y_op_vec = np.tile(yop, self.Hp)

        constraints += [Y_pred >= Y_min_vec - Y_op_vec]
        constraints += [Y_pred <= Y_max_vec - Y_op_vec]

        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status != cp.OPTIMAL and prob.status != cp.OPTIMAL_INACCURATE:
             print(f"Warning: MPC optimization status: {prob.status}")
             # Fallback: keep previous input (dU=0)
             du_opt = np.zeros(nu)
        else:
             # Extract first control move
             du_opt = dU.value[0:nu]

        u_next_dev = u_km1_dev + du_opt
        u_next = u_next_dev + uop

        self.last_u = u_next

        return u_next
