# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import warp as wp

from ..kernels import stable_pd_controller_kernel
from .pd import ActuatorPD


class ActuatorStablePD(ActuatorPD):
    """Stateless stable PD controller (Tan et al. 2011, scalar/per-DOF form).

    Control law:
        τ = clamp(constant + act + Kp·(target_pos - q - q̇·Δt) + Kd·(target_vel - v), ±max_force)

    The ``-q̇·Δt`` term predicts the next-step position (implicit-in-position PD),
    giving better numerical stability under high gains than a standard PD
    controller. Requires ``dt`` to be passed to ``step()``; ``dt == 0`` reduces
    exactly to :class:`ActuatorPD` behaviour.

    Unlike the full Tan 2011 formulation (which uses the mass matrix ``M(q)``
    and inverse dynamics ``G(q, q̇)`` to solve for ``qddot`` implicitly), this
    per-DOF scalar form is compatible with Newton's per-DOF actuator kernel
    architecture and is the variant commonly used in learning-based character
    controllers (DeepMimic, AMP, ASE).

    Reference:
        Tan, J., Liu, K., & Turk, G. (2011). "Stable proportional-derivative
        controllers." IEEE Computer Graphics and Applications, 31(4):34-44.
        DOI: 10.1109/MCG.2011.30
    """

    def _run_controller(
        self,
        sim_state: Any,
        sim_control: Any,
        controller_output: wp.array,
        output_indices: wp.array,
        current_state: Any,
        dt: float,
    ) -> None:
        """Compute stable PD control forces."""
        if dt is None:
            raise ValueError(
                "ActuatorStablePD.step() requires dt (time step) to be provided; got None."
            )

        control_input = None
        if self.control_input_attr is not None:
            control_input = getattr(sim_control, self.control_input_attr, None)

        wp.launch(
            kernel=stable_pd_controller_kernel,
            dim=self.num_actuators,
            inputs=[
                getattr(sim_state, self.state_pos_attr),
                getattr(sim_state, self.state_vel_attr),
                getattr(sim_control, self.control_target_pos_attr),
                getattr(sim_control, self.control_target_vel_attr),
                control_input,
                self.input_indices,
                self.input_indices,
                output_indices,
                self.kp,
                self.kd,
                self.max_force,
                self.constant_force,
                float(dt),
            ],
            outputs=[controller_output],
        )
