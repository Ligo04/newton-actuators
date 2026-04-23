# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Blocked LLT (Cholesky) factorization and solve kernels using Warp's Tile API.

Single-block, dense, 2D-array version for actuator-scale systems. The matrix
size ``n`` and tile edge length ``block_size`` are baked into the kernel at
compile time (``@functools.cache``-memoized per ``(n, block_size)`` pair) so
tile shapes are known to the compiler.

Adapted from ``newton/_src/solvers/kamino/_src/linalg/factorize/llt_blocked.py``
(Newton Developers, Apache-2.0). Key simplifications vs the kamino original:

- **Single block only.** ``num_blocks = 1``. The per-block ``dim`` / ``mio`` /
  ``vio`` index arrays and ``wp.func_native`` pointer-offset helpers are gone.
  Launch with ``wp.launch_tiled(..., dim=[1], block_dim=...)``.
- **2D arrays, no flat layout.** Inputs are plain ``wp.array(ndim=2)``.
- **No in-kernel boundary padding.** When ``n % block_size != 0`` we require
  the caller to pre-pad the storage buffers (``A``, ``L``, ``b``, ``y``, ``x``)
  out to ``n_padded = ⌈n / block_size⌉ · block_size`` with **identity layout**:

      A[:n, :n] = real SPD matrix          L[:n, :n] = Cholesky factor (output)
      A[:n, n:] = 0                        L[:n, n:] = 0
      A[n:, :n] = 0                        L[n:, :n] = 0
      A[n:, n:] = I_{n_padded - n}         L[n:, n:] = I_{n_padded - n}

      b[:n, 0] = real rhs                  y[:n, 0], x[:n, 0] = real output
      b[n:, 0] = 0                         y[n:, 0], x[n:, 0] = 0

  The phantom identity on the trailing diagonal keeps
  ``tile_cholesky_inplace`` / ``tile_*_solve_inplace`` well-conditioned
  (their outputs on the phantom region stay identity). The caller's
  ``build_A`` kernel only needs to write ``A[:n, :n]`` — the zero off-diagonal
  phantom rows/cols and the trailing ``I`` must be established once at
  allocation time (see ``pad_identity_2d``) and are preserved across steps
  because no one writes them.

  This contract avoids the shared-memory visibility issues that an in-kernel
  "conditionally write identity into the tile before cholesky" pass runs
  into with Warp's current tile intrinsics.

- **No in-place solve variant.** The non-in-place ``solve`` is enough for all
  current callers. Can be added later if needed.

Example (``n=50`` system, ``block_size=32`` → ``n_padded=64``)::

    from newton_actuators._src._linalg import (
        make_llt_blocked_factorize_kernel,
        make_llt_blocked_solve_kernel,
        next_block_multiple,
        pad_identity_2d,
    )

    n, block_size = 50, 32
    n_padded = next_block_multiple(n, block_size)  # 64

    # Factory memoizes per block_size — one compiled kernel handles any n.
    factorize_kernel = make_llt_blocked_factorize_kernel(block_size)
    solve_kernel = make_llt_blocked_solve_kernel(block_size)

    A = pad_identity_2d(n, n_padded, device="cuda:0")  # zeros + trailing I
    L = pad_identity_2d(n, n_padded, device="cuda:0")
    b = wp.zeros((n_padded, 1), dtype=wp.float32, device="cuda:0")
    y = wp.zeros((n_padded, 1), dtype=wp.float32, device="cuda:0")
    x = wp.zeros((n_padded, 1), dtype=wp.float32, device="cuda:0")

    # ... caller fills A[:n, :n] and b[:n, 0] each step, e.g. via
    # ... wp.launch(build_A_kernel, dim=(n, n), ...) and
    # ... wp.launch(build_b_kernel, dim=n, ...)

    wp.launch_tiled(factorize_kernel, dim=[1], inputs=[A, L, n], block_dim=128)
    wp.launch_tiled(solve_kernel, dim=[1], inputs=[L, b, y, x, n], block_dim=128)
"""

from functools import cache

import numpy as np
import warp as wp

__all__ = [
    "make_llt_blocked_factorize_kernel",
    "make_llt_blocked_solve_kernel",
    "next_block_multiple",
    "pad_identity_2d",
]


wp.set_module_options({"enable_backward": False})


def next_block_multiple(n: int, block_size: int) -> int:
    """Return the smallest multiple of ``block_size`` that is ``>= n``."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return ((n + block_size - 1) // block_size) * block_size


def pad_identity_2d(n: int, n_padded: int, device: wp.DeviceLike = None) -> wp.array:
    """Allocate a ``(n_padded, n_padded)`` ``float32`` array laid out for the LLT kernels.

    The first ``n`` rows/cols are zero (to be filled by the caller); the
    trailing diagonal block ``[n:, n:]`` is the identity. See the module
    docstring for why this layout is required.
    """
    if n > n_padded:
        raise ValueError(f"n ({n}) must not exceed n_padded ({n_padded})")
    buf = np.zeros((n_padded, n_padded), dtype=np.float32)
    if n_padded > n:
        idx = np.arange(n, n_padded)
        buf[idx, idx] = 1.0
    return wp.array(buf, dtype=wp.float32, device=device)


@cache
def make_llt_blocked_factorize_kernel(block_size: int):
    """Return a blocked Cholesky factorization kernel.

    The returned kernel reads ``A`` (shape ``(n_padded, n_padded)``,
    identity-padded — see module docstring) and writes the lower-triangular
    Cholesky factor ``L`` (same shape, same layout) such that
    ``L[:n,:n] · L[:n,:n]ᵀ = A[:n,:n]`` and ``L[n:,n:] = I``. ``n`` is passed
    to the kernel at launch time as a scalar, so a single compiled kernel
    handles any matrix dimension for a given ``block_size``.

    Only the lower triangle of each block of ``L`` is guaranteed to be
    written; the strict upper half is left in the identity-padded zero state
    set at allocation time (solvers downstream only read the lower triangle).

    The kernel is ``@functools.cache``-memoized on ``block_size`` alone.

    Launch with::

        wp.launch_tiled(kernel, dim=[1], inputs=[A, L, n], block_dim=128)

    Args:
        block_size: Tile edge length. Must be positive. Typical choice: 32.
            ``block_size`` stays a compile-time constant because tile shapes
            must be known to the compiler.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    # ``n`` is passed at launch time so the outer loop bounds are runtime
    # values — this prevents Warp from unrolling them and blowing past per-SM
    # shared memory (each unrolled ``tile_load(storage="shared")`` instance
    # allocates its own shmem slot otherwise).

    @wp.kernel
    def kernel(
        A: wp.array(dtype=float, ndim=2),
        L: wp.array(dtype=float, ndim=2),
        n: int,
    ):
        tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        n_padded = ((n + block_size - 1) // block_size) * block_size

        # Iterate over block-columns along the leading dimension.
        for k in range(0, n_padded, block_size):
            A_kk_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(k, k), storage="shared")

            # Materialize the trailing diagonal tile into shared memory (important for
            # the tile type inference downstream — dropping this block makes the panel
            # loop's ``tile_transpose(A_ik_tile)`` fail type deduction). The caller
            # already pre-padded A[n:, n:] to identity so this write is redundant on
            # values, but we keep it to preserve the intended codegen path.
            if k + block_size > n:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for i in range(num_iterations):
                    linear_index = tid_block + i * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = A_kk_tile[row, col]
                    if k + row >= n or k + col >= n:
                        value = wp.where(row == col, wp.float32(1.0), wp.float32(0.0))
                    A_kk_tile[row, col] = value

            # A_kk -= Σ_{j<k} L_kj · L_kjᵀ (in-place accumulating matmul, α=-1).
            # Transpose is passed inline — naming the intermediate variable makes
            # Warp try to emit ``wp::assign`` for a non-owning view at the loop-end
            # carry, which has no valid overload and breaks NVRTC compile.
            if k > 0:
                for j in range(0, k, block_size):
                    L_kj = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j))
                    wp.tile_matmul(L_kj, wp.tile_transpose(L_kj), A_kk_tile, alpha=-1.0)

            # Cholesky the updated diagonal tile in place → becomes L_kk.
            wp.tile_cholesky_inplace(A_kk_tile)
            wp.tile_store(L, A_kk_tile, offset=(k, k))

            # Process the column panel L_ik for i > k:
            # L_ik = (A_ik − Σ_{j<k} L_ij · L_kjᵀ) · L_kk⁻ᵀ.
            for i in range(k + block_size, n_padded, block_size):
                A_ik_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(i, k), storage="shared")

                # Same tile-materialize trick for the off-diagonal panel tiles.
                if i + block_size > n or k + block_size > n:
                    num_tile_elements = block_size * block_size
                    num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                    for ii in range(num_iterations):
                        linear_index = tid_block + ii * num_threads_per_block
                        linear_index = linear_index % num_tile_elements
                        row = linear_index // block_size
                        col = linear_index % block_size
                        value = A_ik_tile[row, col]
                        if i + row >= n or k + col >= n:
                            value = wp.where(i + row == k + col, wp.float32(1.0), wp.float32(0.0))
                        A_ik_tile[row, col] = value

                if k > 0:
                    for j in range(0, k, block_size):
                        L_ij = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j))
                        L_kj = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j))
                        wp.tile_matmul(L_ij, wp.tile_transpose(L_kj), A_ik_tile, alpha=-1.0)

                # Solve L_kk · L_ikᵀ = A_ikᵀ via the lower-triangular triangular solve.
                # ``tile_lower_solve_inplace`` writes its result into the transposed
                # view, which also mutates ``A_ik_tile`` (the underlying storage);
                # re-transpose ``A_ik_tile`` directly when storing — don't introduce
                # a named intermediate (see matmul-loop note above).
                wp.tile_lower_solve_inplace(A_kk_tile, wp.tile_transpose(A_ik_tile))
                wp.tile_store(L, A_ik_tile, offset=(i, k))

    return kernel


@cache
def make_llt_blocked_solve_kernel(block_size: int):
    """Return a kernel that solves ``L · Lᵀ · x = b`` given the Cholesky factor ``L``.

    Inputs (all shape ``(n_padded, n_padded)`` for ``L`` and ``(n_padded, 1)``
    for ``b`` / ``y`` / ``x``, identity-padded — see module docstring):

    - ``L`` — from :func:`make_llt_blocked_factorize_kernel`
    - ``b`` — real rhs in ``b[:n, 0]``, zeros in ``b[n:, 0]``

    Outputs:

    - ``y`` — forward-sub intermediate, satisfies ``L · y = b``
    - ``x`` — final solution, satisfies ``Lᵀ · x = y`` (so ``L · Lᵀ · x = b``)

    The phantom entries ``y[n:, 0]`` and ``x[n:, 0]`` remain zero because the
    identity padding on ``L[n:, n:]`` propagates zero rhs to zero solution.

    The kernel is ``@functools.cache``-memoized on ``block_size`` alone;
    ``n`` is a runtime kernel arg.

    Launch with::

        wp.launch_tiled(kernel, dim=[1], inputs=[L, b, y, x, n], block_dim=128)
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    # See factorize kernel: ``n`` is passed as a runtime arg to keep the outer
    # loops non-unrolled so shared memory usage stays bounded.

    @wp.kernel
    def kernel(
        L: wp.array(dtype=float, ndim=2),
        b: wp.array(dtype=float, ndim=2),
        y: wp.array(dtype=float, ndim=2),
        x: wp.array(dtype=float, ndim=2),
        n: int,
    ):
        tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        n_padded = ((n + block_size - 1) // block_size) * block_size

        # Forward substitution: solve L · y = b.
        for i in range(0, n_padded, block_size):
            rhs_tile = wp.tile_load(b, shape=(block_size, 1), offset=(i, 0))
            L_diag = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i))
            if i > 0:
                for j in range(0, i, block_size):
                    L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y, shape=(block_size, 1), offset=(j, 0))
                    wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)
            wp.tile_lower_solve_inplace(L_diag, rhs_tile)
            wp.tile_store(y, rhs_tile, offset=(i, 0))

        # Backward substitution: solve Lᵀ · x = y (bottom-up over block-rows).
        for i in range(n_padded - block_size, -1, -block_size):
            i_end = i + block_size
            rhs_tile = wp.tile_load(y, shape=(block_size, 1), offset=(i, 0))
            L_diag = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i))

            # Materialize-and-repair the trailing diagonal tile (see factorize kernel —
            # the redundant writes preserve the tile-type codegen path even though
            # the caller already laid out L[n:, n:] = I externally).
            if i + block_size > n:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for ii in range(num_iterations):
                    linear_index = tid_block + ii * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = L_diag[row, col]
                    if i + row >= n:
                        value = wp.where(i + row == i + col, wp.float32(1.0), wp.float32(0.0))
                    L_diag[row, col] = value

            # Pass transposes inline — see factorize kernel note on loop-end
            # ``wp::assign`` for non-owning views.
            if i_end < n_padded:
                for j in range(i_end, n_padded, block_size):
                    L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(j, i))
                    x_tile = wp.tile_load(x, shape=(block_size, 1), offset=(j, 0))
                    wp.tile_matmul(wp.tile_transpose(L_tile), x_tile, rhs_tile, alpha=-1.0)
            wp.tile_upper_solve_inplace(wp.tile_transpose(L_diag), rhs_tile)
            wp.tile_store(x, rhs_tile, offset=(i, 0))

    return kernel
