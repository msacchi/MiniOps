"""
    hard_threshold(u, tau) -> v

Apply hard-thresholding to an array.

Each element of `u` whose magnitude is smaller than `tau` is set to
zero. Elements with magnitude greater than or equal to `tau` are
left unchanged.

Arguments
---------
u   : Input array (real or complex).
tau : Non-negative threshold parameter.

Returns
-------
v : Array
    Thresholded output of the same shape as `u`.

Notes
-----
- For tau = 0, the input is returned unchanged.
- The function is applied elementwise.
- Unlike soft-thresholding, retained coefficients are not reduced
  in magnitude.
- Works for real or complex arrays.
"""
function hard_threshold(u, tau)
    return ifelse.(abs.(u) .>= tau, u, zero(eltype(u)))
end


"""
    iht(A, y, u0, tau, step_size; niter = 500, verbose = false) -> u

Solve a sparse inverse problem using iterative hard thresholding (IHT).

The method alternates between a gradient descent step for the data
misfit term and a hard-thresholding step that promotes sparsity while
preserving the amplitude of retained coefficients.

Arguments
---------
A         : Linear operator (MiniOps.Op).

y         : Observed data.

u0        : Initial estimate of the solution.

tau       : Hard-threshold parameter.

step_size : Gradient descent step size.

niter     : Number of iterations (default = 500).

verbose   : Print convergence information every 20 iterations.

Returns
-------
u : Array
    Estimated solution after `niter` iterations.

Notes
-----
- The step size should typically be smaller than 1 / ||A||².
- Uses hard_threshold after each gradient step.
- Retained coefficients are not shrunk in amplitude.
- Works for vectors or multidimensional arrays.
- Useful for sparse reconstruction and inverse problems.
"""
function iht(A, y, u0, tau, step_size; niter=500, verbose=false)
    u = copy(u0)

    for k in 1:niter
        # Gradient of 0.5 * ||A*u - y||^2
        r = A * u .- y
        g = A' * r

        # Gradient step on L2 term
        u .= u .- step_size .* g

        # Hard-thresholding step
        u .= hard_threshold(u, tau)

        if verbose && (k % 20 == 0)
            @show k, maximum(abs.(r))
        end
    end

    return u
end


"""
    soft_threshold(u, tau) -> v

Apply soft-thresholding (shrinkage) to an array.

Each element of `u` is shrunk toward zero according to the threshold parameter `tau`. Values with 
magnitude below `tau` are mapped to zero, and larger values are reduced in magnitude.

Arguments
---------
u   : Input array (real or complex).
tau : Non-negative threshold parameter.

Returns
-------
v : Array
    Thresholded output of the same shape as `u`.

Notes
-----
- For tau = 0, the input is returned unchanged.
- The function is applied elementwise.
- Division by zero is avoided internally.
- This is the proximal operator for the L1 norm.
"""
function soft_threshold(u, tau)
    # tau >= 0 (scalar)
    amp    = abs.(u)
    denom  = amp .+ eps(Float64)   # avoid division by zero
    factor = max.(0.0, 1 .- tau ./ denom)
    return factor .* u
end



"""
    ista(A, y, u0, mu, step_size; niter = 100, verbose = false) -> u

Solve an L1-regularized least squares problem using ISTA
(Iterative Shrinkage-Thresholding Algorithm).

The method alternates between a gradient descent step for the data
misfit term and a soft-thresholding step that promotes sparsity.

Arguments
---------
A         : Linear operator (MiniOps.Op).

y         : Observed data.

u0        : Initial estimate of the solution.

mu        : L1 regularization weight.

step_size : Gradient descent step size.

niter     : Number of iterations (default = 100).

verbose   : Print convergence information every 20 iterations.

Returns
-------
u : Array

    Estimated solution after `niter` iterations.

Notes
-----
- The step size should be smaller than 1 / ||A||² for convergence.
- Uses soft_threshold as the proximal operator.
- Works for vectors or multidimensional arrays.
- Often used in sparse reconstruction and inverse problems.
"""
function ista(A, y, u0, mu, step_size; niter=100, verbose=false)
    u = copy(u0)

    for k in 1:niter
        # Gradient of 0.5 * ||A*u - y||^2
        r = A * u .- y
        g = A' * r

        # Gradient step on L2 term
        u .= u .- step_size .* g

        # Prox step (soft-threshold) for L1
        u .= soft_threshold(u, mu * step_size)

        if verbose && (k % 20 == 0)
            @show k, maximum(abs.(r))
        end
    end

    return u
end




"""
    cgls(A, b, mu, x0; tol = 1e-6, max_iter = 1000) -> x

Solve a quadratic regularized least squares problem using the
Conjugate Gradient Least Squares (CGLS) algorithm.

This method minimizes a least-squares objective with Tikhonov-style
regularization by iteratively solving the normal equations using
conjugate gradients.

Arguments
---------
A         : Linear operator or matrix.

b         : Right-hand side vector or array.

mu        : Regularization parameter.

x0        : Initial guess for the solution.

tol       : Convergence tolerance (default = 1e-6).

max_iter  : Maximum number of iterations (default = 1000).

Returns
-------
x : Array
    Estimated solution.

Notes
-----
- Works with both matrices and operator-based solvers.
- Particularly useful for large inverse problems.
- Larger values of mu enforce stronger regularization.
- Convergence is based on the gradient norm.

Example
-------
```julia	
# Decon example
A = conv1d_op(randn(3))
x_true = randn(10);
b = A*x_true 
x = cgls(A, b, 0.01, zeros(size(x_true)))
```
"""
function cgls(A, b, mu, x0; tol=1e-6, max_iter=1000)
    r = b - A * x0           # Initial residual
    s = A' * r - mu * x0     # Negative gradient of regularized objective
    p = copy(s)               # Initial search direction
    old_inner_product = dot(s, s)  # Inner product of s with itself
    x = copy(x0)             # Do not alias the caller's initial guess

    for k in 1:max_iter
        q = A * p            # Compute A*p
   delta = norm(q)^2 + mu * norm(p)^2
        alpha = old_inner_product / delta 
        x += alpha * p       # Update solution
        r -= alpha * q       # Update residual
        s = A' * r - mu * x  # Compute new gradient
        new_inner_product = norm(s)^2  # Inner product of new s with itself
        
        # Check for convergence
        if sqrt(new_inner_product) < tol
            break
        end
        
        beta = new_inner_product / old_inner_product  # Compute beta
        p = s + beta * p            # Update search direction
        old_inner_product = new_inner_product  # Update old inner product
    end

    return x  # Return the computed solution
end


