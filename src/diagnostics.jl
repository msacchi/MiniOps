using LinearAlgebra

"""
    adjoint_test(A::Op, x, y; tol = 1e-10) -> (passed, err)

Check numerically whether an operator `A` satisfies the adjoint
consistency condition using random test vectors.

The function verifies that the forward operator `A` and its adjoint `A'`
are implemented consistently by testing whether

    dot(A*x, y) ≈ dot(x, A'*y)

within a relative tolerance.

Arguments
---------
A   : Linear operator of type `Op`.
x   : Input array in the domain of `A`.
y   : Input array in the range of `A`.
tol : Relative error tolerance for the test (default = 1e-10).

Returns
-------
(passed, err)

passed : Bool
    True if the relative adjoint error is below `tol`.

err : Float64
    The relative adjoint mismatch between the two inner products.

Notes
-----
- Works for vectors or multidimensional arrays.
- Inputs are internally vectorized using `vec`.
- Typical errors should be near machine precision for a correct adjoint.
- This test is essential when developing inverse-problem solvers,
  optimization algorithms, and PDE operators.

Example
-------
```julia 
A = fft_op()
x = randn(64)
y = randn(64)

ok, err = adjoint_test(A, x, y)

println("Passed: ", ok, "   Error: ", err)
```
"""
function adjoint_test(A::Op, x, y; tol=1e-10)
    Ax  = A * x
    Aty = A' * y

    lhs = dot(vec(Ax), vec(y))
    rhs = dot(vec(x),  vec(Aty))

    err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), eps())
    return (err < tol, err)
end


"""
    linearity_test(A::Op, x, z; ntests = 3, tol = 1e-10) -> (passed, err)

Check whether a linear operator `A` satisfies linearity:

    A(a*x + b*z) ≈ a*A(x) + b*A(z)

The test is repeated `ntests` times using random scalars `a` and `b`,
and the worst relative error is reported.

Arguments
---------
A      : Linear operator to test.
x, z   : Input arrays of the same shape and type, in the domain of `A`.
ntests : Number of random trials (default = 3).
tol    : Relative error tolerance.

Returns
-------
(passed, err)

passed : Bool
    True if all tests pass within tolerance.

err : Float64
    Maximum relative error over all trials.

Notes
-----
- Inputs may be real or complex.
- Arrays are vectorized internally using `vec`.
- This test is useful for validating that an operator implementation
  behaves as a true linear map before using it inside solvers.
"""
function linearity_test(A::Op, x, z; ntests=3, tol=1e-10)
    maxerr = 0.0
    for _ in 1:ntests
        a = randn()
        b = randn()

        lhs = A * (a .* x .+ b .* z)
        rhs = a .* (A * x) .+ b .* (A * z)

        err = norm(vec(lhs - rhs)) / max(norm(vec(lhs)), norm(vec(rhs)), eps())
        maxerr = max(maxerr, err)
        err < tol || return (false, maxerr)
    end
    return (true, maxerr)
end


"""
    opnorm_power(A::Op, x0; niter = 20) -> σ

Estimate the spectral norm (largest singular value) of a linear operator
using the power iteration method applied to A' * A.

Starting from an initial vector `x0`, the algorithm repeatedly applies
A' * A and normalizes the result, converging toward the dominant
eigenvalue. The returned value is the square root of that eigenvalue.

Arguments
---------
A     : Linear operator.
x0    : Initial vector (must be in the domain of A).
niter : Number of power iterations (default = 20).

Returns
-------
σ : Float64
    Estimated spectral norm of A.

Notes
-----
- This returns an approximation, not an exact value.
- Convergence depends on separation between singular values.
- Works for real or complex operators.
- Useful for estimating step sizes in optimization algorithms.
"""
function opnorm_power(A::Op, x0; niter=20)
    x = copy(x0)
    x ./= norm(vec(x))

    λ = zero(real(eltype(x)))
    for _ in 1:niter
        y = A' * (A * x)
        λ = norm(vec(y))
        λ == 0 && return 0.0
        x .= y ./ λ
    end

    return sqrt(λ)
end




"""
    is_selfadjoint(A::Op, x; tol = 1e-10) -> (passed, err)

Test whether an operator behaves as self-adjoint (symmetric / Hermitian)
by comparing the action of `A` and `A'` on a test vector.

Arguments
---------
A   : Linear operator.
x   : Test array in the domain of `A`.
tol : Relative error tolerance.

Returns
-------
(passed, err)

passed : Bool
    True if A(x) and A'(x) agree within tolerance.

err : Float64
    Relative discrepancy between A(x) and A'(x).

Notes
-----
- This is a numerical check, not a proof.
- Arrays are vectorized internally using `vec`.
- A positive result indicates that the implementation is consistent
  with a self-adjoint operator for the given input.
"""
function is_selfadjoint(A::Op, x; tol=1e-10)
    lhs = A * x
    rhs = A' * x

    err = norm(vec(lhs - rhs)) / max(norm(vec(lhs)), norm(vec(rhs)), eps(real(float(norm(lhs)))))
    return (err < tol, err)
end



