using Printf

# ---------------------------------------------------------------------
# Internal reporting utilities
# ---------------------------------------------------------------------

"""
    _solver_header(name)

Print a common convergence-table header for iterative solvers.

This is an internal utility used by `iht`, `ista`, and `cgls`.
"""
function _solver_header(name)
    println()
    println(name)
    @printf("%8s %16s %16s %16s\n",
            "iter", "rel_residual", "residual", "cost")
    println(repeat("-", 60))
end


"""
    _solver_report(k, rnorm, rnorm0, cost)

Print one row of solver convergence information.

The reported relative residual is

    rnorm / rnorm0

where `rnorm0` is the norm of the initial residual.
"""
function _solver_report(k, rnorm, rnorm0, cost)
    @printf("%8d %16.6e %16.6e %16.6e\n",
            k, rnorm / rnorm0, rnorm, cost)
end


"""
    hard_threshold(u, tau) -> v

Apply elementwise hard thresholding.

For each element `u[i]`,

- if `abs(u[i]) >= tau`, the value is retained;
- if `abs(u[i]) < tau`, the value is set to zero.

# Arguments

- `u`: Input array. May be real or complex.
- `tau`: Non-negative hard-threshold level.

# Returns

- `v`: Thresholded array with the same shape as `u`.

# Notes

- For `tau = 0`, all entries are retained.
- Hard thresholding does not shrink the amplitude of retained coefficients.
- The operation is applied elementwise.
"""
function hard_threshold(u, tau)
    return ifelse.(abs.(u) .>= tau, u, zero(eltype(u)))
end


"""
    iht(A, y, u0, tau, step_size;
        niter = 500,
        verbose = false,
        print_every = 20) -> u

Solve a sparse inverse problem using Iterative Hard Thresholding (IHT).

At iteration `k`, the method performs a gradient step for the
least-squares data misfit,

    0.5 * ||A*u - y||_2^2,

followed by hard thresholding:

    r = A*u - y
    u = hard_threshold(u - step_size * A' * r, tau)

The threshold `tau` directly controls which coefficients are retained.
Unlike ISTA, IHT does not correspond here to an explicitly added
regularization term in the reported objective. Therefore the reported
`cost` is the data-misfit cost

    cost = 0.5 * ||A*u - y||_2^2.

# Arguments

- `A`: Forward linear operator or matrix. The adjoint must be available as `A'`.
- `y`: Observed data.
- `u0`: Initial model or coefficient array.
- `tau`: Hard-threshold level.
- `step_size`: Gradient-descent step size.

# Keyword arguments

- `niter=500`: Number of IHT iterations.
- `verbose=false`: If `true`, print convergence information.
- `print_every=20`: Print convergence information every `print_every`
  iterations. Iteration 0 and the final iteration are also printed.

# Convergence output

When `verbose=true`, the columns are

- `iter`: Iteration number.
- `rel_residual`: `||r_k||_2 / ||r_0||_2`.
- `residual`: `||r_k||_2`.
- `cost`: `0.5 * ||r_k||_2^2`.

The residual reported for iteration `k` is evaluated after the
iteration-`k` model update.

# Returns

- `u`: Estimated model or coefficient array after `niter` iterations.

# Notes

- A typical stability requirement is

      step_size < 1 / ||A||_2^2.

- The function returns only the solution array, preserving the original
  calling convention.
- Works with vectors and multidimensional arrays as long as `A*u` and
  `A'*r` are defined.

# Example

```julia
u = iht(A, y, zeros(size(u_true)), tau, step_size;
        niter=200, verbose=true, print_every=20)
```
"""
function iht(A, y, u0, tau, step_size;
             niter=500, verbose=false, print_every=20)

    print_every >= 1 ||
        throw(ArgumentError("print_every must be at least 1"))

    u = copy(u0)

    r0 = A * u .- y
    rnorm0_raw = norm(r0)
    rnorm0 = max(rnorm0_raw, eps(Float64))

    if verbose
        _solver_header("IHT")
        cost0 = 0.5 * rnorm0_raw^2
        _solver_report(0, rnorm0_raw, rnorm0, cost0)
    end

    for k in 1:niter
        r = A * u .- y
        g = A' * r

        u .= u .- step_size .* g
        u .= hard_threshold(u, tau)

        if verbose && (k % print_every == 0 || k == niter)
            r_report = A * u .- y
            rnorm = norm(r_report)
            cost = 0.5 * rnorm^2
            _solver_report(k, rnorm, rnorm0, cost)
        end
    end

    return u
end


"""
    soft_threshold(u, tau) -> v

Apply elementwise soft thresholding (shrinkage).

For each element `u[i]`, the magnitude is reduced by `tau` while
preserving phase/sign, and values with magnitude less than or equal
to the threshold are mapped to zero.

Equivalently,

    v[i] = max(0, 1 - tau / abs(u[i])) * u[i].

# Arguments

- `u`: Input array. May be real or complex.
- `tau`: Non-negative soft-threshold level.

# Returns

- `v`: Thresholded array with the same shape as `u`.

# Notes

- For `tau = 0`, the input is unchanged.
- The operation is applied elementwise.
- Division by zero is avoided internally.
- Soft thresholding is the proximal operator of the `L1` norm.
"""
function soft_threshold(u, tau)
    amp = abs.(u)
    denom = amp .+ eps(Float64)
    factor = max.(0.0, 1 .- tau ./ denom)
    return factor .* u
end


"""
    ista(A, y, u0, mu, step_size;
         niter = 100,
         verbose = false,
         print_every = 20) -> u

Solve an `L1`-regularized least-squares problem using ISTA
(Iterative Shrinkage-Thresholding Algorithm).

ISTA minimizes

    cost(u) = 0.5 * ||A*u - y||_2^2 + mu * ||u||_1.

Each iteration consists of a gradient step for the quadratic data
misfit followed by soft thresholding:

    r = A*u - y
    u = soft_threshold(u - step_size * A' * r,
                       mu * step_size)

# Arguments

- `A`: Forward linear operator or matrix. The adjoint must be available as `A'`.
- `y`: Observed data.
- `u0`: Initial model or coefficient array.
- `mu`: `L1` regularization weight.
- `step_size`: Gradient-descent step size.

# Keyword arguments

- `niter=100`: Number of ISTA iterations.
- `verbose=false`: If `true`, print convergence information.
- `print_every=20`: Print convergence information every `print_every`
  iterations. Iteration 0 and the final iteration are also printed.

# Convergence output

When `verbose=true`, the columns are

- `iter`: Iteration number.
- `rel_residual`: `||r_k||_2 / ||r_0||_2`.
- `residual`: `||r_k||_2`.
- `cost`:

      0.5 * ||A*u_k - y||_2^2 + mu * ||u_k||_1.

The residual and cost reported for iteration `k` are evaluated after
the iteration-`k` model update.

# Returns

- `u`: Estimated model or coefficient array after `niter` iterations.

# Notes

- A standard convergence condition is

      step_size < 1 / ||A||_2^2.

- The function returns only the solution array, preserving the original
  calling convention.
- Works with vectors and multidimensional arrays as long as `A*u` and
  `A'*r` are defined.

# Example

```julia
u = ista(A, y, zeros(size(u_true)), mu, step_size;
         niter=200, verbose=true, print_every=20)
```
"""
function ista(A, y, u0, mu, step_size;
              niter=100, verbose=false, print_every=20)

    print_every >= 1 ||
        throw(ArgumentError("print_every must be at least 1"))

    u = copy(u0)

    r0 = A * u .- y
    rnorm0_raw = norm(r0)
    rnorm0 = max(rnorm0_raw, eps(Float64))

    if verbose
        _solver_header("ISTA")
        cost0 = 0.5 * rnorm0_raw^2 + mu * sum(abs, u)
        _solver_report(0, rnorm0_raw, rnorm0, cost0)
    end

    for k in 1:niter
        r = A * u .- y
        g = A' * r

        u .= u .- step_size .* g
        u .= soft_threshold(u, mu * step_size)

        if verbose && (k % print_every == 0 || k == niter)
            r_report = A * u .- y
            rnorm = norm(r_report)
            cost = 0.5 * rnorm^2 + mu * sum(abs, u)
            _solver_report(k, rnorm, rnorm0, cost)
        end
    end

    return u
end


"""
    cgls(A, b, mu, x0;
         tol = 1e-6,
         max_iter = 1000,
         verbose = false,
         print_every = 20) -> x

Solve a quadratically regularized least-squares problem using
Conjugate Gradient Least Squares (CGLS).

The method minimizes the Tikhonov objective

    cost(x) = 0.5 * ||A*x - b||_2^2
            + 0.5 * mu * ||x||_2^2,

which leads to the normal equations

    (A' * A + mu * I) * x = A' * b.

The implementation applies conjugate gradients to these normal
equations without explicitly forming `A' * A`.

# Arguments

- `A`: Forward linear operator or matrix. The adjoint must be available as `A'`.
- `b`: Observed data or right-hand side.
- `mu`: Quadratic (`L2`) regularization parameter.
- `x0`: Initial model.

# Keyword arguments

- `tol=1e-6`: Stopping tolerance on the norm of the regularized
  least-squares gradient,

      ||A'*(b - A*x) - mu*x||_2.

- `max_iter=1000`: Maximum number of CGLS iterations.
- `verbose=false`: If `true`, print convergence information.
- `print_every=20`: Print convergence information every `print_every`
  iterations. Iteration 0, the final iteration, and a converged
  iteration are also printed.

# Convergence output

When `verbose=true`, the columns are

- `iter`: Iteration number.
- `rel_residual`: `||r_k||_2 / ||r_0||_2`, where `r_k = b - A*x_k`.
- `residual`: `||r_k||_2`.
- `cost`:

      0.5 * ||A*x_k - b||_2^2
      + 0.5 * mu * ||x_k||_2^2.

# Returns

- `x`: Estimated solution.

# Notes

- The function returns only the solution array, preserving the original
  calling convention.
- `mu = 0` gives the unregularized least-squares problem.
- The stopping criterion is based on the regularized gradient norm, not
  directly on the data residual.
- Works with matrices and operator-based implementations that define
  both forward and adjoint multiplication.

# Example

```julia
A = conv1d_op(randn(3))
x_true = randn(10)
b = A * x_true

x = cgls(A, b, 0.01, zeros(size(x_true));
         tol=1e-6, max_iter=500,
         verbose=true, print_every=20)
```
"""
function cgls(A, b, mu, x0;
              tol=1e-6, max_iter=1000,
              verbose=false, print_every=20)

    print_every >= 1 ||
        throw(ArgumentError("print_every must be at least 1"))

    x = copy(x0)

    r = b - A * x
    rnorm0_raw = norm(r)
    rnorm0 = max(rnorm0_raw, eps(Float64))

    s = A' * r - mu * x
    p = copy(s)
    old_inner_product = real(dot(s, s))

    if verbose
        _solver_header("CGLS")
        cost0 = 0.5 * rnorm0_raw^2 + 0.5 * mu * norm(x)^2
        _solver_report(0, rnorm0_raw, rnorm0, cost0)
    end

    for k in 1:max_iter
        q = A * p

        delta = norm(q)^2 + mu * norm(p)^2
        alpha = old_inner_product / delta

        x .+= alpha .* p
        r .-= alpha .* q

        s = A' * r - mu * x
        new_inner_product = real(dot(s, s))
        grad_norm = sqrt(new_inner_product)

        converged = grad_norm < tol

        if verbose &&
           (k % print_every == 0 || converged || k == max_iter)
            rnorm = norm(r)
            cost = 0.5 * rnorm^2 + 0.5 * mu * norm(x)^2
            _solver_report(k, rnorm, rnorm0, cost)
        end

        if converged
            break
        end

        beta = new_inner_product / old_inner_product
        p .= s .+ beta .* p
        old_inner_product = new_inner_product
    end

    return x
end
