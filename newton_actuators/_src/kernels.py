# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp


@wp.func
def _interp_1d(
    x: float,
    xs: wp.array(dtype=float),
    ys: wp.array(dtype=float),
    n: int,
) -> float:
    """Linearly interpolate (x -> y) from sorted sample arrays, clamping at boundaries."""
    if n <= 0:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]
    for k in range(n - 1):
        if xs[k + 1] >= x:
            dx = xs[k + 1] - xs[k]
            if dx == 0.0:
                return ys[k]
            t = (x - xs[k]) / dx
            return ys[k] + t * (ys[k + 1] - ys[k])
    return ys[n - 1]


@wp.kernel
def pd_controller_kernel(
    current_pos: wp.array(dtype=float),
    current_vel: wp.array(dtype=float),
    target_pos: wp.array(dtype=float),
    target_vel: wp.array(dtype=float),
    control_input: wp.array(dtype=float),
    state_indices: wp.array(dtype=wp.uint32),
    target_indices: wp.array(dtype=wp.uint32),
    output_indices: wp.array(dtype=wp.uint32),
    kp: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    max_force: wp.array(dtype=float),
    constant_force: wp.array(dtype=float),
    # Optional DC motor velocity-dependent saturation (pass None to skip):
    saturation_effort: wp.array(dtype=float),
    velocity_limit: wp.array(dtype=float),
    # Optional angle-dependent torque lookup (pass None/0 to skip):
    lookup_angles: wp.array(dtype=float),
    lookup_torques: wp.array(dtype=float),
    lookup_size: int,
    output: wp.array(dtype=float),
):
    """Unified PD controller with optional DC motor saturation and angle-dependent limits.

    Force: f = constant + act + kp*(target_pos - q) + kd*(target_vel - v)

    Clipping (in priority order):
        - DC motor:  velocity-dependent τ_min/τ_max from saturation_effort and velocity_limit
        - Lookup:    angle-dependent ±limit interpolated from lookup table
        - Box:       ±max_force (fallback)

    Result is added to output.
    """
    i = wp.tid()
    state_idx = state_indices[i]
    target_idx = target_indices[i]
    out_idx = output_indices[i]

    position_error = target_pos[target_idx] - current_pos[state_idx]
    velocity_error = target_vel[target_idx] - current_vel[state_idx]

    const_f = float(0.0)
    if constant_force:
        const_f = constant_force[i]

    act = float(0.0)
    if control_input:
        act = control_input[target_idx]

    force = const_f + act + kp[i] * position_error + kd[i] * velocity_error

    if saturation_effort:
        vel = current_vel[state_idx]
        sat = saturation_effort[i]
        vel_lim = velocity_limit[i]
        max_f = max_force[i]
        max_torque = wp.clamp(sat * (1.0 - vel / vel_lim), 0.0, max_f)
        min_torque = wp.clamp(sat * (-1.0 - vel / vel_lim), -max_f, 0.0)
        force = wp.clamp(force, min_torque, max_torque)
    elif lookup_size > 0:
        torque_limit = _interp_1d(current_pos[state_idx], lookup_angles, lookup_torques, lookup_size)
        force = wp.clamp(force, -torque_limit, torque_limit)
    else:
        force = wp.clamp(force, -max_force[i], max_force[i])

    output[out_idx] = output[out_idx] + force


@wp.kernel
def stable_pd_build_rhs_kernel(
    current_pos: wp.array(dtype=float),
    current_vel: wp.array(dtype=float),
    target_pos: wp.array(dtype=float),
    target_vel: wp.array(dtype=float),
    state_indices: wp.array(dtype=wp.uint32),
    target_indices: wp.array(dtype=wp.uint32),
    kp: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    bias_forces: wp.array(dtype=float),
    dt: float,
    b: wp.array(dtype=float),
):
    """Assemble the Tan 2011 rhs vector per actuator:

        b[i] = kp[i]·(target_pos - q - q̇·dt) + kd[i]·(target_vel - q̇) - C[i]

    One thread per actuator (``dim=num_actuators``). Writes the full rhs
    consumed by the host-side ``(M + diag(kd)·dt)·qddot = b`` solve in
    ``ActuatorStablePD._run_controller``.
    """
    i = wp.tid()
    state_idx = state_indices[i]
    target_idx = target_indices[i]

    q_i = current_pos[state_idx]
    qd_i = current_vel[state_idx]
    qt_i = target_pos[target_idx]
    qdt_i = target_vel[target_idx]

    p_term_i = kp[i] * (qt_i - q_i - qd_i * dt)
    d_term_i = kd[i] * (qdt_i - qd_i)

    c_i = float(0.0)
    if bias_forces:
        c_i = bias_forces[i]

    b[i] = p_term_i + d_term_i - c_i


@wp.kernel
def stable_pd_build_A_kernel(
    mass_matrix: wp.array(dtype=float, ndim=2),
    kd: wp.array(dtype=float),
    dt: float,
    A: wp.array(dtype=float, ndim=2),
):
    """Assemble ``A[:n, :n] = M + diag(kd)·dt`` for the Tan 2011 implicit solve.

    Launched with ``dim=(n, n)``. One thread per matrix entry, no ``for``.
    ``A`` is expected to be the ``(n_padded, n_padded)`` scratch buffer
    allocated via ``_linalg.pad_identity_2d`` — this kernel only writes the
    real ``[:n, :n]`` region; the identity-padded phantom region ``[n:, n:]``
    is established once at allocation time and preserved across steps.
    """
    i, j = wp.tid()
    a_ij = mass_matrix[i, j]
    if i == j:
        a_ij = a_ij + kd[i] * dt
    A[i, j] = a_ij


@wp.kernel
def stable_pd_controller_kernel(
    current_pos: wp.array(dtype=float),
    current_vel: wp.array(dtype=float),
    target_pos: wp.array(dtype=float),
    target_vel: wp.array(dtype=float),
    control_input: wp.array(dtype=float),
    state_indices: wp.array(dtype=wp.uint32),
    target_indices: wp.array(dtype=wp.uint32),
    output_indices: wp.array(dtype=wp.uint32),
    kp: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    max_force: wp.array(dtype=float),
    constant_force: wp.array(dtype=float),
    qddot: wp.array(dtype=float),
    dt: float,
    output: wp.array(dtype=float),
):
    """Tan 2011 step 5 only — compute τ given the pre-solved implicit acceleration ``qddot``::

        τ = clamp(const + act + kp·(target_pos - q - q̇·dt)
                              + kd·(target_vel - q̇)
                              - kd·qddot·dt,  ±max_force)

    ``qddot`` must already satisfy ``(M + diag(kd)·dt)·qddot = p_term + d_term - C``;
    populate it by launching :func:`stable_pd_build_rhs_kernel` and then solving
    the dense system on the host (see ``ActuatorStablePD._run_controller``).

    One thread per actuator (``dim=num_actuators``). Adds to ``output``.

    Reference: Tan, J., Liu, K., & Turk, G. (2011). "Stable proportional-
    derivative controllers." IEEE Computer Graphics and Applications,
    31(4):34-44. DOI: 10.1109/MCG.2011.30
    """
    i = wp.tid()
    state_idx = state_indices[i]
    target_idx = target_indices[i]
    out_idx = output_indices[i]

    q_i = current_pos[state_idx]
    qd_i = current_vel[state_idx]
    qt_i = target_pos[target_idx]
    qdt_i = target_vel[target_idx]

    p_term_i = kp[i] * (qt_i - q_i - qd_i * dt)
    d_term_i = kd[i] * (qdt_i - qd_i)

    const_f = float(0.0)
    if constant_force:
        const_f = constant_force[i]
    act = float(0.0)
    if control_input:
        act = control_input[target_idx]

    tau_i = const_f + act + p_term_i + d_term_i - kd[i] * qddot[i] * dt
    tau_i = wp.clamp(tau_i, -max_force[i], max_force[i])

    output[out_idx] = output[out_idx] + tau_i


@wp.kernel
def nn_output_kernel(
    nn_torques: wp.array(dtype=float),
    max_force: wp.array(dtype=float),
    output_indices: wp.array(dtype=wp.uint32),
    output: wp.array(dtype=float),
):
    """Clamp neural-network output torques to ±max_force and add to controller output."""
    i = wp.tid()
    out_idx = output_indices[i]
    force = wp.clamp(nn_torques[i], -max_force[i], max_force[i])
    output[out_idx] = output[out_idx] + force


@wp.kernel
def pid_controller_kernel(
    current_pos: wp.array(dtype=float),
    current_vel: wp.array(dtype=float),
    target_pos: wp.array(dtype=float),
    target_vel: wp.array(dtype=float),
    control_input: wp.array(dtype=float),
    state_indices: wp.array(dtype=wp.uint32),
    target_indices: wp.array(dtype=wp.uint32),
    output_indices: wp.array(dtype=wp.uint32),
    kp: wp.array(dtype=float),
    ki: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    max_force: wp.array(dtype=float),
    integral_max: wp.array(dtype=float),
    constant_force: wp.array(dtype=float),
    dt: float,
    current_integral: wp.array(dtype=float),
    output: wp.array(dtype=float),
):
    """PID control with anti-windup: f = clamp(constant + act + kp*(target_pos - q) + ki*integral + kd*(target_vel - v), ±max_force). Adds to output."""
    i = wp.tid()
    state_idx = state_indices[i]
    target_idx = target_indices[i]
    out_idx = output_indices[i]

    position_error = target_pos[target_idx] - current_pos[state_idx]
    velocity_error = target_vel[target_idx] - current_vel[state_idx]

    integral = current_integral[i] + position_error * dt
    integral = wp.clamp(integral, -integral_max[i], integral_max[i])

    const_f = float(0.0)
    if constant_force:
        const_f = constant_force[i]

    act = float(0.0)
    if control_input:
        act = control_input[target_idx]

    force = const_f + act + kp[i] * position_error + ki[i] * integral + kd[i] * velocity_error
    force = wp.clamp(force, -max_force[i], max_force[i])

    output[out_idx] = output[out_idx] + force


@wp.kernel
def pid_integral_state_kernel(
    current_pos: wp.array(dtype=float),
    target_pos: wp.array(dtype=float),
    state_indices: wp.array(dtype=wp.uint32),
    target_indices: wp.array(dtype=wp.uint32),
    integral_max: wp.array(dtype=float),
    dt: float,
    current_integral: wp.array(dtype=float),
    next_integral: wp.array(dtype=float),
):
    """Update PID integral state with anti-windup."""
    i = wp.tid()
    state_idx = state_indices[i]
    target_idx = target_indices[i]

    position_error = target_pos[target_idx] - current_pos[state_idx]

    integral = current_integral[i] + position_error * dt
    integral = wp.clamp(integral, -integral_max[i], integral_max[i])

    next_integral[i] = integral


@wp.kernel
def delay_buffer_state_kernel(
    target_pos_global: wp.array(dtype=float),
    target_vel_global: wp.array(dtype=float),
    control_input_global: wp.array(dtype=float),
    indices: wp.array(dtype=wp.uint32),
    copy_idx: int,
    write_idx: int,
    current_buffer_pos: wp.array2d(dtype=float),
    current_buffer_vel: wp.array2d(dtype=float),
    current_buffer_act: wp.array2d(dtype=float),
    next_buffer_pos: wp.array2d(dtype=float),
    next_buffer_vel: wp.array2d(dtype=float),
    next_buffer_act: wp.array2d(dtype=float),
):
    """Update delay circular buffer: copy missing entry, write new entry."""
    i = wp.tid()
    global_idx = indices[i]

    next_buffer_pos[copy_idx, i] = current_buffer_pos[copy_idx, i]
    next_buffer_vel[copy_idx, i] = current_buffer_vel[copy_idx, i]
    next_buffer_act[copy_idx, i] = current_buffer_act[copy_idx, i]

    next_buffer_pos[write_idx, i] = target_pos_global[global_idx]
    next_buffer_vel[write_idx, i] = target_vel_global[global_idx]

    act = float(0.0)
    if control_input_global:
        act = control_input_global[global_idx]
    next_buffer_act[write_idx, i] = act
