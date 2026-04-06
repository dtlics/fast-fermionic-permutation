import pennylane as qp
import numpy as np


class CNotPowGate(qp.operation.Operator):
    num_wires = 2
    num_params = 1

    def __init__(self, t: float, wires: qp.wires.WiresLike):
        self.exponent = t
        super().__init__(t, wires=wires, id=None)

    def compute_matrix(self, t):
        g = qp.math.exp((1j * np.pi * t) / 2)
        s = qp.math.sin(np.pi * t / 2)
        c = qp.math.cos(np.pi * t / 2)
        return qp.math.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, g*c, -1j*g*s],
            [0, 0, -1j*g*s, g*c]
        ])


class ZPowGate(qp.operation.Operator):
    num_wires = 1
    num_params = 2

    def __init__(self, t: float, s: float, wires: qp.wires.WiresLike):
        self.exponent = t
        super().__init__(t, s, wires=wires, id=None)

    def compute_matrix(self, t, s):
        a = qp.math.exp(1j * np.pi * s * t)
        b = qp.math.exp(1j * np.pi * t)
        return a * qp.math.array([
            [1, 0],
            [0, b]
        ])


class CZPowGate(qp.operation.Operator):
    num_wires = 2
    num_params = 1

    def __init__(self, t: float, wires: qp.wires.WiresLike):
        self.exponent = t
        super().__init__(t, wires=wires, id=None)

    def compute_matrix(self, t):
        g = qp.math.exp(1j * np.pi * t)
        return qp.math.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, g]
        ])


class FSwapPowGate(qp.operation.Operator):
    num_wires = 2
    num_params = 2

    def __init__(self, t: float, s: float, wires: qp.wires.WiresLike):
        self.exponent = t
        super().__init__(t, s, wires=wires, id=None)

    def compute_matrix(self, t):
        p = qp.math.exp(1j * np.pi * t)
        g = qp.math.exp((1j * np.pi * t) / 2)
        s = qp.math.sin(np.pi * t / 2)
        c = qp.math.cos(np.pi * t / 2)
        return qp.math.array([
            [1, 0, 0, 0],
            [0, 0, g*c, -1j*g*s],
            [0, -1j*g*s, g*c, 0],
            [0, 0, 0, p]
        ])


class FSWAP(qp.operation.Operator):
    num_params=0
    num_wires=2

    def __init__(self, wires: qp.wires.WiresLike):
        self.exponent = 1
        super().__init__(wires)

    def compute_matrix(self, t):
        p = qp.math.exp(1j * np.pi)
        g = qp.math.exp((1j * np.pi) / 2)
        s = qp.math.sin(np.pi / 2)
        c = qp.math.cos(np.pi / 2)
        return qp.math.array([
            [1, 0, 0, 0],
            [0, g * c, -1j * g * s, 0],
            [0, -1j * g * s, g * c, 0],
            [0, 0, 0, p]
        ])
