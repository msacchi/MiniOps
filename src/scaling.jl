"""
    scaling_op(alpha::Number) -> Op

Create a linear operator that multiplies any input array by scalar `alpha`.
The adjoint multiplies by `conj(alpha)`, so the operator is a correct
Hermitian linear operator for both real and complex scalars.

The input shape is preserved and `(m,n)=(-1,-1)` because the size is not
fixed at construction.
"""
function scaling_op(alpha::Number)
    f = x -> alpha .* x
    ft = y -> conj(alpha) .* y
    return Op(f, ft; m=-1, n=-1, name=:scaling)
end

"""
    diag_op(w) -> Op

Create an elementwise diagonal operator

```julia
y = w .* x
```

with adjoint `conj.(w) .* y`. `w` may be a vector or multidimensional array
and must be broadcast-compatible with the input.
"""
function diag_op(w)
    wcopy = copy(w)
    f = x -> wcopy .* x
    ft = y -> conj.(wcopy) .* y
    return Op(f, ft; m=-1, n=-1, name=:diag)
end
