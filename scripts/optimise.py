from scipy.optimize import minimize, LinearConstraint
import numpy as np
import pandas as pd
import os

from tontine.portfolio import objective
from tontine.payoffs import Context
from tontine.jacobian import DerivativeHelper, jac_factory

survival_curve = pd.read_excel(os.path.join("data", "AG2022prob.xlsx"))
context = Context(0.85, 1e5, 0.02, survival_curve, 10, 0.04, 0.10, 0.15, 0.5)
helper = DerivativeHelper(context)

jac = jac_factory(context, helper)

res = minimize(
    lambda x: -objective(x, context),
    np.array([0.1, 0.1, 0.4, 0.4]),
    method="trust-constr",
    constraints=[
        LinearConstraint(np.eye(4), np.zeros(4)),
        LinearConstraint(np.ones((4,)), 1, 1),
    ],
    jac=jac,
    tol=1e-20,
    options={"disp": True, "maxiter": int(1e4), "initial_constr_penalty": 10},
)

print(res)
