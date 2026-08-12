import casadi as ca
import numpy as np


def _scaled_expression(variable, scale):
    """Convert normalized decision variables to physical quantities."""
    return ca.repmat(ca.DM(scale).reshape((-1, 1)), 1, variable.size2()) * variable


def _add_bounds(opti, variable, lower, upper, scale):
    """Apply physical bounds to normalized variables, column-wise."""
    scale = np.asarray(scale, dtype=float).reshape((-1, 1))
    lower = np.asarray(lower, dtype=float).reshape((-1, 1)) / scale
    upper = np.asarray(upper, dtype=float).reshape((-1, 1)) / scale
    ncols = variable.size2()
    opti.subject_to(opti.bounded(
        ca.repmat(ca.DM(lower), 1, ncols),
        variable,
        ca.repmat(ca.DM(upper), 1, ncols),
    ))


def _add_physical_bounds(opti, expression, lower, upper, scale):
    """Apply bounds to an expression whose values are already physical."""
    scale = np.asarray(scale, dtype=float).reshape((-1, 1))
    lower = np.asarray(lower, dtype=float).reshape((-1, 1))
    upper = np.asarray(upper, dtype=float).reshape((-1, 1))
    ncols = expression.size2()
    scale = ca.repmat(ca.DM(scale), 1, ncols)
    lower = ca.repmat(ca.DM(lower), 1, ncols)
    upper = ca.repmat(ca.DM(upper), 1, ncols)
    opti.subject_to(opti.bounded(lower / scale, expression / scale, upper / scale))


def _set_initial(opti, variable, values, scale):
    """Set an initial value on a normalized variable matrix."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        if variable.size1() == 1:
            values = values.reshape((1, -1))
        else:
            values = values.reshape((-1, 1))
    scale = np.asarray(scale, dtype=float)
    if scale.size == 1:
        values = values / scale.item()
    else:
        values = values / scale.reshape((-1, 1))
    opti.set_initial(variable, values)


def createOptiProblem(model):
    """Create a direct collocation NLP with globally allocated trajectories.

    States and controls retain the original transcription: both have values at
    every mesh endpoint and collocation point, and controls retain the
    collocation interpolation/continuity constraints.  The difference is that
    all variables are allocated once, outside the interval loop.  Interval
    slices are then used to assemble the collocation equations and outputs.
    """
    opti = ca.Opti()

    nx = model.states.num_x
    nu = model.controls.num_u
    ng = model.parameters.num_g
    n_intervals = model.mesh_numIntervals
    degree = model.collocation_degree
    n_grid = n_intervals * (degree + 1) + 1

    # Normalized decision variables are allocated globally.  Physical
    # expressions are derived once and passed to the model function.
    X_nodes_bar = opti.variable(nx, n_intervals + 1)
    X_colloc_bar = opti.variable(nx, n_intervals * degree)
    U_nodes_bar = opti.variable(nu, n_intervals + 1)
    U_colloc_bar = opti.variable(nu, n_intervals * degree)

    X_nodes = _scaled_expression(X_nodes_bar, model.states.scale)
    X_colloc = _scaled_expression(X_colloc_bar, model.states.scale)
    U_nodes = _scaled_expression(U_nodes_bar, model.controls.scale)
    U_colloc = _scaled_expression(U_colloc_bar, model.controls.scale)

    _add_bounds(opti, X_nodes_bar, model.states.lb, model.states.ub, model.states.scale)
    _add_bounds(opti, X_colloc_bar, model.states.lb, model.states.ub, model.states.scale)
    _add_bounds(opti, U_nodes_bar, model.controls.lb, model.controls.ub, model.controls.scale)
    _add_bounds(opti, U_colloc_bar, model.controls.lb, model.controls.ub, model.controls.scale)

    # Parameters are also allocated once. They are not NLP decision variables.
    Gs = opti.parameter(ng, n_grid)
    if ng > 0:
        parameter_values = np.vstack([
            np.asarray(value, dtype=float).reshape(1, -1)
            for value in model.parameters.value
        ])
        if parameter_values.shape[1] != n_grid:
            raise ValueError(
                "Parameter initial values must have one value per mesh/collocation point "
                f"({n_grid}); received {parameter_values.shape[1]}."
            )
        opti.set_value(Gs, parameter_values)

    # Reconstruct the original public trajectory ordering:
    # endpoint, collocation points, endpoint, collocation points, ..., endpoint.
    X_columns = [X_nodes[:, 0]]
    U_columns = [U_nodes[:, 0]]
    G_columns = [Gs[:, 0]]
    for interval in range(n_intervals):
        colloc_start = interval * degree
        colloc_end = (interval + 1) * degree
        X_columns.extend([X_colloc[:, colloc_start:colloc_end][:, j] for j in range(degree)])
        U_columns.extend([U_colloc[:, colloc_start:colloc_end][:, j] for j in range(degree)])
        G_columns.extend([Gs[:, 1 + interval * (degree + 1) + j] for j in range(degree)])
        X_columns.append(X_nodes[:, interval + 1])
        U_columns.append(U_nodes[:, interval + 1])
        G_columns.append(Gs[:, 1 + interval * (degree + 1) + degree])

    Xs = ca.horzcat(*X_columns)
    Us = ca.horzcat(*U_columns)
    Gs_ordered = ca.horzcat(*G_columns)

    # Set initial guesses in the globally allocated variables.  The public
    # ordering above is used to map the supplied initial mesh trajectory.
    # Initial profiles are supplied by model implementations and may be
    # either (n_grid,) or (n_grid, 1), depending on the interpolation source.
    # Normalize each profile independently rather than converting the whole
    # list to one NumPy array.
    state_initial = [np.asarray(profile, dtype=float).reshape(-1) for profile in model.states.x_init]
    for i in range(nx):
        _set_initial(opti, X_nodes_bar[i, :], state_initial[i][[0] + [1 + k * (degree + 1) + degree for k in range(n_intervals)]], model.states.scale[i])
        colloc_indices = [1 + k * (degree + 1) + j for k in range(n_intervals) for j in range(degree)]
        _set_initial(opti, X_colloc_bar[i, :], state_initial[i][colloc_indices], model.states.scale[i])

        if model.states.BC[i] == 4:
            opti.subject_to(X_nodes_bar[i, 0] == model.states.BCini[i] / model.states.scale[i])
            opti.subject_to(X_nodes_bar[i, -1] == model.states.BCend[i] / model.states.scale[i])
        elif model.states.BC[i] == 3:
            opti.subject_to(X_nodes_bar[i, 0] == X_nodes_bar[i, -1])
        elif model.states.BC[i] == 2:
            opti.subject_to(X_nodes_bar[i, -1] == model.states.BCend[i] / model.states.scale[i])
        elif model.states.BC[i] == 1:
            opti.subject_to(X_nodes_bar[i, 0] == model.states.BCini[i] / model.states.scale[i])
        elif model.states.BC[i] != 0:
            raise ValueError("Boundary Conditions are Incorrectly Defined")

    control_initial = [np.asarray(profile, dtype=float).reshape(-1) for profile in model.controls.u_init]
    for i in range(nu):
        _set_initial(opti, U_nodes_bar[i, :], control_initial[i][[0] + [1 + k * (degree + 1) + degree for k in range(n_intervals)]], model.controls.scale[i])
        colloc_indices = [1 + k * (degree + 1) + j for k in range(n_intervals) for j in range(degree)]
        _set_initial(opti, U_colloc_bar[i, :], control_initial[i][colloc_indices], model.controls.scale[i])

    cost = 0
    for interval in range(n_intervals):
        colloc_start = interval * degree
        colloc_end = (interval + 1) * degree
        Xk = X_nodes[:, interval]
        Xc = X_colloc[:, colloc_start:colloc_end]
        Uk = U_nodes[:, interval]
        Uc = U_colloc[:, colloc_start:colloc_end]
        Gc = Gs[:, 1 + interval * (degree + 1):1 + interval * (degree + 1) + degree]

        rhs, L, path_constraints, _ = model.modelFunction(Xc, Uc, Gc)

        if path_constraints.size1() > 0:
            _add_physical_bounds(
                opti,
                path_constraints,
                model.path_constraints.lb,
                model.path_constraints.ub,
                model.path_constraints.scale,
            )

        cost += ca.mtimes(L, ca.DM(model.collocation_B * model.mesh_size))

        Z_s = ca.horzcat(Xk, Xc)
        Z_u = ca.horzcat(Uk, Uc)
        Pidot = (1 / model.mesh_size) * ca.mtimes(Z_s, ca.DM(model.collocation_C))
        state_scale_inv = ca.diag(1 / ca.DM(model.states.scale))
        opti.subject_to(
            ca.mtimes(state_scale_inv, Pidot)
            == ca.mtimes(state_scale_inv, rhs)
        )

        Xk_end = ca.mtimes(Z_s, ca.DM(model.collocation_D))
        Uk_end = ca.mtimes(Z_u, ca.DM(model.collocation_D))
        opti.subject_to(Xk_end / ca.DM(model.states.scale) == X_nodes_bar[:, interval + 1])
        opti.subject_to(Uk_end / ca.DM(model.controls.scale) == U_nodes_bar[:, interval + 1])

    opti.minimize(cost)
    return opti, Xs, Us, Gs_ordered
