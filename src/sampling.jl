"""
    sampling_op(idx, n)
    sampling_op(idx, (n,))
    sampling_op(idx, full_size)

Construct a sampling operator.

Two sampling conventions are supported.

# 1. Scalar/vector sampling

For a vector of length `n`,

```julia
S = sampling_op(idx, n)
y = S * x
```

returns `x[idx]`. The adjoint `S' * y` scatters the samples back into a
length-`n` vector. Repeated indices are allowed and are accumulated by the
adjoint.

The equivalent tuple form `sampling_op(idx, (n,))` is also supported.

# 2. Whole-trace sampling

For an array whose first dimension is time,

```julia
full_size = (nt, n2, n3, ...)
S = sampling_op(idx, full_size)
```

all dimensions after time are flattened into a trace index using Julia's
column-major ordering. If

```julia
ntr = prod(full_size[2:end])
X = reshape(x, nt, ntr)
```

then the forward operation is exactly

```julia
S * x == X[:, idx]
```

and therefore returns an `nt × length(idx)` matrix. The adjoint scatters
those complete traces back into an array of shape `full_size`, accumulating
contributions when `idx` contains duplicates.

For clarity, the same whole-trace operation can be constructed explicitly
with [`trace_sampling_op`](@ref).

# Index convention

For `full_size = (nt, n2, n3, ...)`, `idx` ranges from
`1:prod(full_size[2:end])`. For example, for `(nt, nx, ny)`, the spatial
location `(ix, iy)` corresponds to

```julia
j = LinearIndices((nx, ny))[ix, iy]
```

# Examples

```julia
# 1D samples
S = sampling_op([1, 5, 9], 10)
y = S * randn(10)                  # length 3

# Whole traces from a 2D gather
X = randn(1000, 120)
S = sampling_op([1, 20, 75], size(X))
D = S * X                          # size (1000, 3)
Xadj = S' * D                      # size (1000, 120)

# Whole traces from a 4D array: (time, x, y, component)
X4 = randn(500, 20, 15, 3)
S4 = trace_sampling_op([1, 17, 50], size(X4))
D4 = S4 * X4                       # size (500, 3)
X4adj = S4' * D4                   # size(X4adj) == size(X4)
```
"""
sampling_op(idx::AbstractVector{<:Integer}, n::Integer) =
    sampling_op(idx, (Int(n),))

function sampling_op(
    idx::AbstractVector{<:Integer},
    full_size::NTuple{N,Int},
) where {N}
    N >= 1 || throw(ArgumentError("full_size must contain at least one dimension"))
    all(s -> s > 0, full_size) || throw(ArgumentError("all entries of full_size must be positive"))

    if N == 1
        idx_vec = Int.(idx)
        n = full_size[1]
        _check_sampling_indices(idx_vec, n, "vector")
        ns = length(idx_vec)

        f = function (x)
            length(x) == n || throw(DimensionMismatch("sampling_op expected $n input elements, got $(length(x))"))
            return reshape(x, n)[idx_vec]
        end

        ft = function (d)
            length(d) == ns || throw(DimensionMismatch("sampling_op adjoint expected $ns data elements, got $(length(d))"))
            D = reshape(d, ns)
            x = zeros(eltype(D), n)
            @inbounds for k in eachindex(idx_vec)
                x[idx_vec[k]] += D[k]
            end
            return x
        end

        return Op(f, ft; m=ns, n=n, name=:sampling)
    end

    return trace_sampling_op(idx, full_size)
end


"""
    trace_sampling_op(idx, full_size)

Construct an operator that samples complete traces from an N-dimensional
array with **time in the first dimension**.

`full_size` must have at least two dimensions:

```julia
(nt, n2)
(nt, n2, n3)
(nt, n2, n3, n4)
...
```

The spatial dimensions `2:end` are flattened to `ntr = prod(full_size[2:end])`.
Forward application reshapes the input to `nt × ntr` and returns
`X[:, idx]`, an `nt × length(idx)` matrix. The adjoint returns an array of
shape `full_size` and sums contributions for repeated trace indices.

This explicit constructor is recommended when code should make it obvious
that sampling acts on **whole traces**, not on individual scalar samples.

See also [`sampling_op`](@ref).
"""
function trace_sampling_op(
    idx::AbstractVector{<:Integer},
    full_size::NTuple{N,Int},
) where {N}
    N >= 2 || throw(ArgumentError("trace_sampling_op requires full_size with at least two dimensions: (nt, n2, ...)"))
    all(s -> s > 0, full_size) || throw(ArgumentError("all entries of full_size must be positive"))

    idx_vec = Int.(idx)
    ns = length(idx_vec)
    nt = full_size[1]
    ntr = prod(full_size[2:end])
    _check_sampling_indices(idx_vec, ntr, "trace")

    f = function (x)
        length(x) == prod(full_size) ||
            throw(DimensionMismatch("trace_sampling_op expected input size $full_size ($(prod(full_size)) elements), got size $(size(x))"))
        X = reshape(x, nt, ntr)
        return X[:, idx_vec]
    end

    ft = function (d)
        length(d) == nt * ns ||
            throw(DimensionMismatch("trace_sampling_op adjoint expected $(nt * ns) data elements, got $(length(d))"))
        D = reshape(d, nt, ns)
        X = zeros(eltype(D), nt, ntr)
        @inbounds for k in eachindex(idx_vec)
            @views X[:, idx_vec[k]] .+= D[:, k]
        end
        return reshape(X, full_size)
    end

    return Op(f, ft; m=nt * ns, n=prod(full_size), name=:trace_sampling)
end


"""
    _check_sampling_indices(idx, n, what)

Validate that every sampling index lies in `1:n`. Empty index vectors are
allowed and define a valid zero-row sampling operator.
"""
function _check_sampling_indices(idx::AbstractVector{<:Integer}, n::Integer, what::AbstractString)
    bad = findfirst(i -> i < 1 || i > n, idx)
    isnothing(bad) && return nothing
    throw(BoundsError(1:n, idx[bad]))
end
