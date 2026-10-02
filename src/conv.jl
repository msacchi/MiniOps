"""
    forward_conv(x, h)

Compute the full 1D discrete convolution of vectors `x` and `h`. The output
has length `length(x) + length(h) - 1` and uses the promoted element type of
`x` and `h`.
"""
function forward_conv(x::AbstractVector, h::AbstractVector)
    nx = length(x)
    nh = length(h)
    T = promote_type(eltype(x), eltype(h))
    y = zeros(T, nx + nh - 1)

    @inbounds for ix in 1:nx, ih in 1:nh
        y[ix + ih - 1] += x[ix] * h[ih]
    end
    return y
end

"""
    adjoint_conv(y, h)

Apply the Hermitian adjoint, with respect to the first argument, of
`forward_conv(x, h)`. For complex filters this uses `conj(h)`.
"""
function adjoint_conv(y::AbstractVector, h::AbstractVector)
    ny = length(y)
    nh = length(h)
    nx = ny - nh + 1
    nx >= 0 || throw(DimensionMismatch("data length must be at least length(h)-1"))
    T = promote_type(eltype(y), eltype(h))
    x = zeros(T, nx)

    @inbounds for ix in 1:nx, ih in 1:nh
        x[ix] += y[ix + ih - 1] * conj(h[ih])
    end
    return x
end

"""
    conv1d_op(h) -> Op

Return a matrix-free full 1D convolution operator with fixed filter `h`.
For an input vector of length `nx` and filter length `nh`, the forward output
has length `nx + nh - 1`. The adjoint is the exact Hermitian transpose of
the convolution map.

```julia
h = [1.0, -0.5, 0.25]
C = conv1d_op(h)
y = C * x
xadj = C' * y
```

The input length is not known at construction, so `size(C) == (-1,-1)`.
Use [`with_shape`](@ref) if fixed flattened dimensions are required.
"""
function conv1d_op(h::AbstractVector)
    hvec = collect(h)
    isempty(hvec) && throw(ArgumentError("h cannot be empty"))
    f = x -> forward_conv(x, hvec)
    ft = y -> adjoint_conv(y, hvec)
    return Op(f, ft; m=-1, n=-1, name=:conv1d)
end

"""
    forward_conv_cols(X, h)

Apply full 1D convolution with filter `h` independently to every column of
matrix `X`. If `size(X) == (nt,ntr)`, the output size is
`(nt + length(h) - 1, ntr)`.
"""
function forward_conv_cols(X::AbstractMatrix, h::AbstractVector)
    nt, ntr = size(X)
    nh = length(h)
    T = promote_type(eltype(X), eltype(h))
    Y = zeros(T, nt + nh - 1, ntr)

    @inbounds for itr in 1:ntr, it in 1:nt, ih in 1:nh
        Y[it + ih - 1, itr] += X[it, itr] * h[ih]
    end
    return Y
end

"""
    adjoint_conv_cols(Y, h)

Apply the Hermitian adjoint of [`forward_conv_cols`](@ref), independently
for each column. Complex filters use conjugated coefficients.
"""
function adjoint_conv_cols(Y::AbstractMatrix, h::AbstractVector)
    ny, ntr = size(Y)
    nh = length(h)
    nt = ny - nh + 1
    nt >= 0 || throw(DimensionMismatch("data length must be at least length(h)-1"))
    T = promote_type(eltype(Y), eltype(h))
    X = zeros(T, nt, ntr)

    @inbounds for itr in 1:ntr, it in 1:nt, ih in 1:nh
        X[it, itr] += Y[it + ih - 1, itr] * conj(h[ih])
    end
    return X
end

"""
    conv1d_cols_op(h) -> Op

Return a full 1D convolution operator that acts independently on the columns
of a matrix. The first dimension is interpreted as the sample/time axis and
the second dimension as traces/channels.

```julia
C = conv1d_cols_op(h)
Y = C * X
Xadj = C' * Y
```
"""
function conv1d_cols_op(h::AbstractVector)
    hvec = collect(h)
    isempty(hvec) && throw(ArgumentError("h cannot be empty"))
    f = X -> forward_conv_cols(X, hvec)
    ft = Y -> adjoint_conv_cols(Y, hvec)
    return Op(f, ft; m=-1, n=-1, name=:conv1d_cols)
end
