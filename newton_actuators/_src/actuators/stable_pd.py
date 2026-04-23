# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import warp as wp

from .._linalg import (
    make_llt_blocked_factorize_kernel,
    make_llt_blocked_solve_kernel,
    next_block_multiple,
    pad_identity_2d,
)
from ..kernels import (
    stable_pd_build_A_kernel,
    stable_pd_build_rhs_kernel,
    stable_pd_controller_kernel,
)
from .pd import ActuatorPD


class ActuatorStablePD(ActuatorPD):
    """Stable PD controller (Tan et al. 2011) — full implicit matrix form.

    Implements the algorithm of bullet3's ``PDControllerStableMultiDof``::

        p_term = Kp · (target_pos - q - q̇·Δt)
        d_term = Kd · (target_vel - v)
        (M + diag(Kd)·Δt) · qddot = p_term + d_term - C
        τ = clamp(const + act + p_term + d_term - Kd·qddot·Δt, ±max_force)

    The ``qddot`` unknown is solved **implicitly** on the GPU via a blocked
    Cholesky factorization of ``A = M + diag(Kd)·Δt`` (see
    ``newton_actuators._src._linalg``). The entire step — rhs assembly,
    ``A``-matrix assembly, factorize, solve, and per-actuator torque
    composition — is executed as a sequence of ``wp.launch`` / ``wp.launch_tiled``
    calls with no host↔device transfers, making the whole pipeline
    CUDA-graph capturable.

    The caller is responsible for populating ``state.mass_matrix`` and
    ``state.bias_forces`` each step from the physics engine (``newton.sim``,
    MuJoCo, PhysX, ...) — this class makes no assumption about the query API.
    Scratch buffers (``A``, ``L``, ``b``, ``y``, ``qddot``) are allocated by
    :meth:`state` with the identity-padded layout required by the tile
    kernels (see ``_linalg.pad_identity_2d``).

    ``step()`` requires both a valid ``State`` and ``dt`` — unlike the scalar
    PD controllers this class does not have a stateless fallback; the physics
    engine's mass-matrix query is mandatory.

    Reference:
        Tan, J., Liu, K., & Turk, G. (2011). "Stable proportional-derivative
        controllers." IEEE Computer Graphics and Applications, 31(4):34-44.
        DOI: 10.1109/MCG.2011.30

        Blocked Cholesky kernels adapted from Newton's kamino LLT blocked
        solver; see ``newton_actuators._src._linalg`` for details.
    """

    @dataclass
    class State:
        """Inputs and scratch buffers for the Tan 2011 solve.

        User-filled each step (populate before :meth:`step`):
            mass_matrix (wp.array): ``M(q)`` effective mass/inertia for this
                actuator subsystem. Shape ``(N, N)``, ``float32``. Obtain via
                ``physics.calculateMassMatrix(...)`` or equivalent.
            bias_forces (wp.array): ``C(q, q̇)`` Coriolis + gravity + external
                force compensation. Shape ``(N,)``, ``float32``. Obtain via
                ``physics.calculateInverseDynamics(q, qdot, 0)``.

        Internal scratch (allocated by :meth:`ActuatorStablePD.state`, not
        intended as user input but exposed for debugging):
            A (wp.array): Augmented mass matrix ``M + diag(Kd)·dt``, stored in
                identity-padded ``(N_padded, N_padded)`` layout. The phantom
                region ``[N:, N:]`` is the identity and is preserved across
                steps (only ``[:N, :N]`` is rewritten each call).
            L (wp.array): Cholesky factor of ``A``. Same shape / layout as ``A``.
            b (wp.array): Rhs ``p_term + d_term - C``. Shape ``(N_padded, 1)``
                with ``b[:N, 0]`` real and ``b[N:, 0] = 0``.
            y (wp.array): Forward-substitution intermediate ``L · y = b``.
                Shape ``(N_padded, 1)``.
            qddot (wp.array): Solution of ``Lᵀ · qddot = y`` (and hence the
                implicit-acceleration system). Shape ``(N_padded, 1)``; the
                per-actuator step-5 kernel reads ``qddot[:N, 0]`` through a
                ``reshape((N_padded,))`` zero-copy 1D view.
        """

        mass_matrix: wp.array = None
        bias_forces: wp.array = None
        A: wp.array = None
        L: wp.array = None
        b: wp.array = None
        y: wp.array = None
        qddot: wp.array = None

        def reset(self) -> None:
            """Zero all allocated buffers in-place.

            Note: ``A`` and ``L`` lose their identity padding on ``n_padded > n``
            after ``reset()`` — callers that care about this invariant should
            re-initialize via :meth:`ActuatorStablePD.state` rather than
            reusing a reset state across solver invocations.
            """
            for arr in (self.mass_matrix, self.bias_forces, self.A, self.L, self.b, self.y, self.qddot):
                if arr is not None:
                    arr.zero_()

    # Default tile edge length for the blocked Cholesky. 32 matches kamino's
    # LLTBlockedSolver default and is a good balance for Tensor-Core fp32
    # throughput on Ampere+ architectures.
    DEFAULT_BLOCK_SIZE: int = 32

    # Launch-time thread-block size for the tile kernels. Must be a multiple
    # of 32 (warp size) and large enough to cover a ``block_size × block_size``
    # tile; 128 works for ``block_size`` up to 32 and matches kamino.
    DEFAULT_TILE_BLOCK_DIM: int = 128

    def __init__(
        self,
        *args: Any,
        block_size: int = DEFAULT_BLOCK_SIZE,
        tile_block_dim: int = DEFAULT_TILE_BLOCK_DIM,
        **kwargs: Any,
    ) -> None:
        """Initialize the stable-PD actuator.

        All positional and keyword arguments except ``block_size`` and
        ``tile_block_dim`` are forwarded to :class:`ActuatorPD.__init__`.

        Args:
            block_size: Tile edge length for the blocked Cholesky (compile-time
                constant baked into the kernels). Default 32.
            tile_block_dim: CUDA thread-block size for the tile kernels.
                Default 128.
        """
        super().__init__(*args, **kwargs)
        self._block_size = int(block_size)
        self._tile_block_dim = int(tile_block_dim)
        self._n_padded = next_block_multiple(self.num_actuators, self._block_size)
        # The factories are @cache-memoized on block_size, so repeat actuators
        # with the same block_size share compiled kernels.
        self._factorize_kernel = make_llt_blocked_factorize_kernel(self._block_size)
        self._solve_kernel = make_llt_blocked_solve_kernel(self._block_size)

    def is_stateful(self) -> bool:
        return True

    def _run_controller(
        self,
        sim_state: Any,
        sim_control: Any,
        controller_output: wp.array,
        output_indices: wp.array,
        current_state: Any,
        dt: float,
    ) -> None:
        """Compute Tan 2011 stable PD forces via a full on-device pipeline."""
        if dt is None:
            raise ValueError("ActuatorStablePD.step() requires dt (time step) to be provided; got None.")
        if current_state is None:
            raise ValueError(
                "ActuatorStablePD.step() requires a State with mass_matrix and bias_forces populated; "
                "got current_act_state=None."
            )
        if current_state.mass_matrix is None:
            raise ValueError("ActuatorStablePD.State.mass_matrix must be populated each step.")
        if current_state.bias_forces is None:
            raise ValueError("ActuatorStablePD.State.bias_forces must be populated each step.")
        if (
            current_state.A is None
            or current_state.L is None
            or current_state.b is None
            or current_state.y is None
            or current_state.qddot is None
        ):
            raise ValueError(
                "ActuatorStablePD.State is missing scratch buffers (A, L, b, y, qddot). "
                "Allocate via actuator.state() rather than constructing State() directly."
            )

        control_input = None
        if self.control_input_attr is not None:
            control_input = getattr(sim_control, self.control_input_attr, None)

        pos = getattr(sim_state, self.state_pos_attr)
        vel = getattr(sim_state, self.state_vel_attr)
        target_pos = getattr(sim_control, self.control_target_pos_attr)
        target_vel = getattr(sim_control, self.control_target_vel_attr)
        dt_f = float(dt)
        n = self.num_actuators

        # The tile kernels expect 2D column vectors of shape (n_padded, 1);
        # build_rhs and step-5 operate on 1D views. ``wp.array.reshape`` is
        # a zero-copy view so this does not break graph capture.
        b_1d = current_state.b.reshape(self._n_padded)
        qddot_1d = current_state.qddot.reshape(self._n_padded)

        # Step 1: b[:n] = p_term + d_term - C (per-actuator, parallel).
        wp.launch(
            kernel=stable_pd_build_rhs_kernel,
            dim=n,
            inputs=[
                pos,
                vel,
                target_pos,
                target_vel,
                self.input_indices,
                self.input_indices,
                self.kp,
                self.kd,
                current_state.bias_forces,
                dt_f,
                b_1d,
            ],
        )

        # Step 2: A[:n, :n] = M + diag(kd)·dt (per-entry, parallel).
        wp.launch(
            kernel=stable_pd_build_A_kernel,
            dim=(n, n),
            inputs=[
                current_state.mass_matrix,
                self.kd,
                dt_f,
                current_state.A,
            ],
        )

        # Step 3: blocked Cholesky factorize A → L (tile-parallel on GPU).
        wp.launch_tiled(
            kernel=self._factorize_kernel,
            dim=[1],
            inputs=[current_state.A, current_state.L, n],
            block_dim=self._tile_block_dim,
        )

        # Step 4: forward + backward substitution → qddot (tile-parallel on GPU).
        wp.launch_tiled(
            kernel=self._solve_kernel,
            dim=[1],
            inputs=[current_state.L, current_state.b, current_state.y, current_state.qddot, n],
            block_dim=self._tile_block_dim,
        )

        # Step 5: τ = const + act + p_term + d_term - kd·qddot·dt, clamp, accumulate.
        wp.launch(
            kernel=stable_pd_controller_kernel,
            dim=n,
            inputs=[
                pos,
                vel,
                target_pos,
                target_vel,
                control_input,
                self.input_indices,
                self.input_indices,
                output_indices,
                self.kp,
                self.kd,
                self.max_force,
                self.constant_force,
                qddot_1d,
                dt_f,
            ],
            outputs=[controller_output],
        )

    def state(self) -> "ActuatorStablePD.State":
        """Return a new state with zero-initialised M, C, and identity-padded solve scratch."""
        device = self.input_indices.device
        n = self.num_actuators
        n_padded = self._n_padded
        return ActuatorStablePD.State(
            mass_matrix=wp.zeros((n, n), dtype=wp.float32, device=device),
            bias_forces=wp.zeros(n, dtype=wp.float32, device=device),
            A=pad_identity_2d(n, n_padded, device=device),
            L=pad_identity_2d(n, n_padded, device=device),
            b=wp.zeros((n_padded, 1), dtype=wp.float32, device=device),
            y=wp.zeros((n_padded, 1), dtype=wp.float32, device=device),
            qddot=wp.zeros((n_padded, 1), dtype=wp.float32, device=device),
        )
