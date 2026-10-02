#----------------------------------------------------------
# Seismic blending operator
#
# A common-receiver gather D(t, s) is blended by delaying
# each source trace according to a prescribed source time
# and summing the delayed traces into one blended record.
#
# Forward:
#
#     b = B * D
#
# where D has size (nt, ns) and b has length ntout.
#
# Adjoint:
#
#     Da = B' * b
#
# where Da has size (nt, ns).  The adjoint is the exact
# transpose of the delay-and-sum operation used in B.
#
# Fractional source times are handled by linear interpolation.
# Source times are relative delays in seconds and must be >= 0.
#----------------------------------------------------------

"""
    forward_blending(D, source_times, dt, ntout)

Blend a common-receiver gather `D` of size `nt × ns` using source firing
times `source_times` (seconds). Each source trace is delayed and summed.
Fractional sample delays are represented with linear interpolation.
"""
function forward_blending(
    D::AbstractMatrix,
    source_times::AbstractVector{<:Real},
    dt::Real,
    ntout::Int,
)
    nt, ns = size(D)

    length(source_times) == ns ||
        throw(ArgumentError("length(source_times) must equal the number of source traces"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    ntout > 0 || throw(ArgumentError("ntout must be positive"))
    all(t -> t >= 0, source_times) ||
        throw(ArgumentError("source_times must contain nonnegative relative delays"))

    T = promote_type(eltype(D), Float64)
    b = zeros(T, ntout)

    @inbounds for is in 1:ns
        shift = source_times[is] / dt
        ishift = floor(Int, shift)
        frac = shift - ishift
        w0 = 1 - frac
        w1 = frac

        for it in 1:nt
            jt = it + ishift

            if 1 <= jt <= ntout
                b[jt] += w0 * D[it, is]
            end

            if w1 != 0 && 1 <= jt + 1 <= ntout
                b[jt + 1] += w1 * D[it, is]
            end
        end
    end

    return b
end


"""
    adjoint_blending(b, source_times, dt, nt)

Adjoint of `forward_blending`. It gathers a blended record `b` back into
an `nt × ns` common-receiver gather using exactly the transpose interpolation
weights of the forward operator.
"""
function adjoint_blending(
    b::AbstractVector,
    source_times::AbstractVector{<:Real},
    dt::Real,
    nt::Int,
)
    ns = length(source_times)
    ntout = length(b)

    dt > 0 || throw(ArgumentError("dt must be positive"))
    nt > 0 || throw(ArgumentError("nt must be positive"))
    all(t -> t >= 0, source_times) ||
        throw(ArgumentError("source_times must contain nonnegative relative delays"))

    T = promote_type(eltype(b), Float64)
    D = zeros(T, nt, ns)

    @inbounds for is in 1:ns
        shift = source_times[is] / dt
        ishift = floor(Int, shift)
        frac = shift - ishift
        w0 = 1 - frac
        w1 = frac

        for it in 1:nt
            jt = it + ishift

            if 1 <= jt <= ntout
                D[it, is] += w0 * b[jt]
            end

            if w1 != 0 && 1 <= jt + 1 <= ntout
                D[it, is] += w1 * b[jt + 1]
            end
        end
    end

    return D
end


"""
    blending_op(source_times, dt, nt; ntout=nothing)

Return a MiniOps blending operator for a common-receiver gather.

The model is an `nt × ns` matrix whose columns are source traces, with
`ns = length(source_times)`. The entries of `source_times` are relative
source firing times in seconds.

Forward:

    b = B * D

delays and sums the source traces into a single blended record.

Adjoint:

    Da = B' * b

gathers the blended record back into an `nt × ns` matrix using the exact
transpose of the forward interpolation.

If `ntout` is omitted, it is chosen large enough to contain the latest
delayed trace:

    ntout = nt + ceil(maximum(source_times) / dt)

Examples
--------

    source_times = [0.0, 0.37, 0.81]
    B = blending_op(source_times, dt, nt)

    b  = B * D
    Da = B' * b
"""
function blending_op(
    source_times::AbstractVector{<:Real},
    dt::Real,
    nt::Int;
    ntout::Union{Nothing,Int} = nothing,
)
    ts = collect(source_times)
    ns = length(ts)

    ns > 0 || throw(ArgumentError("source_times cannot be empty"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    nt > 0 || throw(ArgumentError("nt must be positive"))
    all(t -> t >= 0, ts) ||
        throw(ArgumentError("source_times must contain nonnegative relative delays"))

    nblend = isnothing(ntout) ? nt + ceil(Int, maximum(ts) / dt) : ntout
    nblend > 0 || throw(ArgumentError("ntout must be positive"))

    f = D -> begin
        size(D, 1) == nt ||
            throw(ArgumentError("input gather must have nt rows"))
        size(D, 2) == ns ||
            throw(ArgumentError("input gather must have length(source_times) columns"))
        forward_blending(D, ts, dt, nblend)
    end

    ft = b -> begin
        length(b) == nblend ||
            throw(ArgumentError("blended record must have length ntout"))
        adjoint_blending(vec(b), ts, dt, nt)
    end

    return Op(f, ft; m = nblend, n = nt * ns, name = :blending)
end
