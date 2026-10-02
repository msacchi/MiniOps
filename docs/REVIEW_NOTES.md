# MiniOps review notes — 2026-10-01

## Sampling

The intended contract is now explicit:

- `sampling_op(idx, n)` or `sampling_op(idx, (n,))`: sample scalar entries of a 1-D vector.
- `trace_sampling_op(idx, (nt, n2, n3, ...))`: preserve the first dimension (time), flatten dimensions `2:end`, and sample complete traces.
- `sampling_op(idx, full_size)` with `length(full_size) >= 2` delegates to `trace_sampling_op` for backward compatibility.
- The forward output of trace sampling is always `nt × length(idx)`.
- The adjoint returns the original `full_size` and accumulates repeated trace indices.
- Tests now cover 2-D, 3-D, 4-D, and 5-D arrays plus duplicate indices.

## Problems found and corrected

1. The old 2-D sampling test generated indices over all `nx*ny` scalar elements, but the new operator interprets dimension 1 as preserved time and only allows indices over the flattened trace dimensions. The test was stale.
2. `scaling_op` and `diag_op` did not conjugate complex weights in their adjoints. Corrected.
3. Convolution adjoints did not conjugate a complex filter, and allocation could inherit the wrong element type when `x` and `h` had different real/complex types. Corrected.
4. `cgls` did not include `-mu*x0` in the initial regularized gradient for a nonzero initial guess. Corrected, and the initial guess is now copied.
5. `radon_tx_tp_chirp_adjoint` had a default `f2=60.0` whereas the forward default was `Inf`. Made consistent (`Inf`).
6. Parabolic Radon docstrings referred to obsolete `_lin` routine names. Corrected.
7. `docs/operators.html` and the README were stale relative to the current exports. Updated.

## Important convention, not a bug

The NUDFT implementation uses a unitary FFT convention. At zero coordinate perturbation,
`nudft1_op(zeros(N), dx) * X` equals `sqrt(N) * ifft(X)`, not raw `ifft(X)`.
This explains a factor-of-`sqrt(N)` mismatch when comparing directly with Julia's `ifft`.

## Files that appear legacy or accidental

- `src/radon_linear.jl` is an older duplicate of the parabolic Radon code. It is not included by `MiniOps.jl` and is not part of the active API.
- `src/.fftops.jl.swp` is an editor swap file.
- `src/map` is a text tree listing, not Julia source.
- `__MACOSX` is archive metadata.

These are excluded from the cleaned review archive where appropriate.

## Validation limitation

Julia is not installed in the review environment, so `Pkg.test()` could not be executed here. The source and tests were reviewed statically; run `Pkg.test()` locally under Julia 1.12.x before tagging a release.
