import numpy as np
import control

class LinModel:
    def __init__(self, sys, Ts=None, i_u=None, i_d=None):
        """
        Construct a linear model from state-space model `sys` with sampling time `Ts`.

        Args:
            sys: control.StateSpace object or tuple of matrices (A, B, C, D).
            Ts: Sampling time. If sys is discrete, Ts can be None (uses sys.dt).
            i_u: Indices of manipulated inputs (0-based).
            i_d: Indices of measured disturbances (0-based).
        """
        # Convert to control.StateSpace
        if isinstance(sys, (tuple, list)):
             # Assuming sys is (A, B, C, D)
             sys = control.ss(*sys)

        # Determine inputs
        n_inputs = sys.B.shape[1]
        if i_u is None:
            if i_d is None:
                # Default: all inputs are manipulated
                i_u = list(range(n_inputs))
                i_d = []
            else:
                # Remaining are manipulated
                i_u = [i for i in range(n_inputs) if i not in i_d]
        elif i_d is None:
             # Remaining are disturbances
             i_d = [i for i in range(n_inputs) if i not in i_u]

        self.i_u = np.array(i_u, dtype=int)
        self.i_d = np.array(i_d, dtype=int)

        self.nu = len(self.i_u)
        self.nd = len(self.i_d)

        # Check if continuous
        if sys.dt is None or sys.dt == 0:
            if Ts is None:
                 raise ValueError("Sampling time Ts must be specified for continuous systems")
            self.Ts = Ts
            # Discretize
            # Simple approach: zoh for all
            sys_d = control.c2d(sys, Ts, method='zoh')
            self.A = sys_d.A
            self.B = sys_d.B
            self.C = sys_d.C
            self.D = sys_d.D
        else:
             if Ts is not None and not np.isclose(Ts, sys.dt):
                 print(f"Warning: Resampling from {sys.dt} to {Ts} not implemented, using {sys.dt}")
             self.Ts = sys.dt
             self.A = sys.A
             self.B = sys.B
             self.C = sys.C
             self.D = sys.D

        self.Bu = self.B[:, self.i_u]
        if self.nd > 0:
            self.Bd = self.B[:, self.i_d]
            self.Dd = self.D[:, self.i_d]
        else:
            self.Bd = np.zeros((self.A.shape[0], 0))
            self.Dd = np.zeros((self.C.shape[0], 0))

        # Check strict propriety for u
        Du = self.D[:, self.i_u]
        if not np.allclose(Du, 0):
             raise ValueError("LinModel only supports strictly proper systems (D=0 for manipulated inputs)")

        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]

        # Operating points
        self.uop = np.zeros(self.nu)
        self.yop = np.zeros(self.ny)
        self.dop = np.zeros(self.nd)
        self.xop = np.zeros(self.nx)
        self.fop = np.zeros(self.nx)

        self.x0 = np.zeros(self.nx)

    def setop(self, uop=None, yop=None, dop=None, xop=None, fop=None):
        """Set operating points."""
        if uop is not None: self.uop = np.array(uop)
        if yop is not None: self.yop = np.array(yop)
        if dop is not None: self.dop = np.array(dop)
        if xop is not None: self.xop = np.array(xop)
        if fop is not None: self.fop = np.array(fop)
        return self
