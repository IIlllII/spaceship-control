import Optizelle
import autograd.numpy as np
from autograd import elementwise_grad
from autograd import hessian
from autograd import jacobian
from autograd import make_hvp


# Returns an array with three values. It is assumed that you are turning from t=0
# The array contains:
# [time to start accelerating and turning, time to stop turning and only accelerate, total time until stop]
#
# Current limitations are:
# - This does not handle cases with counter clockwise turns
# - Does not currently not handle cases with non 180 degree turns
# These limitations are trivial to fix though.
def full_stop(acceleration, turn_rate, d0, v0):

    # create needed derivative functions
    first_deriv = elementwise_grad(objective)
    hessian_vp = make_hvp(objective)

    jacobian_x = jacobian(constraint_x, 0)
    jacobian_y = jacobian(constraint_y, 0)

    hs_x = hessian(constraint_x, 0)
    hs_y = hessian(constraint_y, 0)

    # Calculate initial guess
    ta = np.pi / turn_rate
    tm = np.linalg.norm(v0) / acceleration

    # Generate an initial guess
    x = np.array([ta, ta, ta + tm])

    # Initial dual
    y = np.array([0.0, 0.0])

    # Create an optimization state
    state = Optizelle.EqualityConstrained.State.t(Optizelle.Rm, Optizelle.Rm, x, y)

    # Read the optimization parameters from file
    Optizelle.json.EqualityConstrained.read(Optizelle.Rm, Optizelle.Rm, "newton_cg.json", state)

    # Create the bundle of functions
    fns = Optizelle.EqualityConstrained.Functions.t()
    fns.f = MyObj(first_deriv, hessian_vp)
    fns.g = MyEq(jacobian_x, jacobian_y, hs_x, hs_y, acceleration, turn_rate, d0, v0)

    # Solve the optimization problem
    Optizelle.EqualityConstrained.Algorithms.getMin(
        Optizelle.Rm, Optizelle.Rm, Optizelle.Messaging.stdout, fns, state)

    # Print out the reason for convergence
    print "The algorithm converged due to: %s" % (
        Optizelle.OptimizationStop.to_string(state.opt_stop))

    # Constraint in x dim
    return state.x


# Constraint that must be satisfied in x direction
def constraint_x(t, acc, s, d0, v0x):
    ts_sum = (acc * np.sin(d0 + s * t[1])) / s
    ta_sum = (acc * np.sin(d0 + s * t[0])) / s

    stop_dir = np.cos(d0 + s * t[1])

    ts_sum_sec = acc * (t[1]) * stop_dir
    tm_sum_sec = acc * (t[2]) * stop_dir

    return v0x + (ts_sum - ta_sum) + (tm_sum_sec - ts_sum_sec)


# Constraint that must be satisfied in y direction
def constraint_y(t, acc, s, d0, v0y):
    ts_sum = -(acc * np.cos(d0 + s * t[1])) / s
    ta_sum = -(acc * np.cos(d0 + s * t[0])) / s

    stop_dir = np.sin(d0 + s * t[1])

    ts_sum_sec = acc * (t[1]) * stop_dir
    tm_sum_sec = acc * (t[2]) * stop_dir

    return v0y + (ts_sum - ta_sum) + (tm_sum_sec - ts_sum_sec)


# The objective function. Only tm (time to stop) is important, so that is what is minimized.
def objective(t):
    return t[2] * t[2]


# Optizelle objective function API
class MyObj(Optizelle.ScalarValuedFunction):

    def __init__(self, first_deriv, hessian_vp):
        self.first_deriv = first_deriv
        self.hessian_vp = hessian_vp

    def eval(self, x):
        obj = objective(x)
        return obj

    def grad(self, x, dx):
        deriv = self.first_deriv(x)
        np.copyto(dx, deriv)

    def hessvec(self, x, dx, H_dx):
        hvp = self.hessian_vp(x)[0]
        mul = hvp(dx)
        np.copyto(H_dx, mul)


# Optizelle equality constraints API
class MyEq(Optizelle.VectorValuedFunction):

    def __init__(self, jacobian_x, jacobian_y, hessian_x, hessian_y, acceleration, turn_rate, d0, v0):
        self.jacobian_x = jacobian_x
        self.jacobian_y = jacobian_y
        self.hessian_x = hessian_x
        self.hessian_y = hessian_y
        self.acceleration = acceleration
        self.turn_rate = turn_rate
        self.d0 = d0
        self.v0 = v0

    # y=g(x)
    def eval(self, x, y):
        y[0] = constraint_x(x, self.acceleration, self.turn_rate, self.d0, self.v0[0])
        y[1] = constraint_y(x, self.acceleration, self.turn_rate, self.d0, self.v0[1])

    # y=g'(x)dx
    def p(self, x, dx, y):
        res_x = self.jacobian_x(x, self.acceleration, self.turn_rate, self.d0, self.v0[0])
        res_y = self.jacobian_y(x, self.acceleration, self.turn_rate, self.d0, self.v0[1])

        x_jac = res_x[0] * dx[0] + res_x[1] * dx[1] + res_x[2] * dx[2]
        y_jac = res_y[0] * dx[0] + res_y[1] * dx[1] + res_y[2] * dx[2]

        y[0] = x_jac
        y[1] = y_jac

    # xhat=g'(x)*dy
    def ps(self, x, dy, xhat):
        res_x = self.jacobian_x(x, self.acceleration, self.turn_rate, self.d0, self.v0[0])
        res_y = self.jacobian_y(x, self.acceleration, self.turn_rate, self.d0, self.v0[1])
        xhat[0] = res_x[0] * dy[0] + res_y[0] * dy[1]
        xhat[1] = res_x[1] * dy[0] + res_y[1] * dy[1]
        xhat[2] = res_x[2] * dy[0] + res_y[2] * dy[1]

    # xhat=(g''(x)dx)*dy
    def pps(self, x, dx, dy, xhat):
        hessian_x = self.hessian_x(x, self.acceleration, self.turn_rate, self.d0, self.v0[0]) * dy[0]
        hessian_y = self.hessian_y(x, self.acceleration, self.turn_rate, self.d0, self.v0[1]) * dy[1]

        tst_x = hessian_x.dot(dx)
        tst_y = hessian_y.dot(dx)

        xhat[0] = tst_x[0] + tst_y[0]
        xhat[1] = tst_x[1] + tst_y[1]
        xhat[2] = tst_x[2] + tst_y[2]


solution = full_stop(2.0, np.pi / 2.0, 0.0, np.array([2.0, 0.0]))

# Print out the final answer
print("The optimal point is:" + str(solution))
