#----------------------------------------------------------
# Forward parabolic Radon, linear time interpolation
#   (τ, q)  →  (t, h)
#----------------------------------------------------------

"""
    radon_tx_forward!(d, m, dt, h, q; href = one(T))

In-place forward parabolic Radon transform (τ–q → t–h) with linear
time interpolation.

Model kinematics:
    t = τ + q * (h / href)^2

Inputs:
  m    :: Matrix{T}          (nt × nq)   model in (τ, q)
  dt   :: T                  time sampling (seconds)
  h    :: Vector{T}          offsets, length nh
  q    :: Vector{T}          Radon parameters, length nq
  href :: T (keyword)        reference offset (must be ≠ 0)

On entry:
  d    :: Matrix{T}          (nt × nh)   (will be overwritten)

On exit:
  d    contains data in (t, h).
"""
function radon_tx_forward!(
    d::AbstractMatrix{T},
    m::AbstractMatrix{T},
    dt::T,
    h::AbstractVector{T},
    q::AbstractVector{T};
    href::T = one(T),
) where {T<:AbstractFloat}

    nt, nq = size(m)
    nh     = length(h)

    @assert size(d, 1) == nt
    @assert size(d, 2) == nh
    @assert href != zero(T)

    fill!(d, zero(T))

    invdt = inv(dt)

    @inbounds for ih in 1:nh
        hr  = h[ih] / href
        hr2 = hr * hr
        for iq in 1:nq
            shift = q[iq] * hr2 * invdt      # = (q * (h/href)^2) / dt
            for itau in 1:nt
                # continuous index x in [1, nt]:
                # x = t/dt + 1 = itau + shift
                x  = itau + shift
                it = floor(Int, x)

                if it >= 1 && it < nt
                    α   = x - T(it)          # fractional part
                    val = m[itau, iq]
                    d[it,   ih] += (one(T) - α) * val
                    d[it+1, ih] += α * val
                end
            end
        end
    end

    return d
end

"""
    radon_tx_forward(m, dt, h, q; href = one(T))

Allocating version of the forward parabolic Radon transform.

Returns:
  d :: Matrix{T}  (nt × nh)  data in (t, h)
"""
function radon_tx_forward(
    m::AbstractMatrix{T},
    dt::T,
    h::AbstractVector{T},
    q::AbstractVector{T};
    href::T = one(T),
) where {T<:AbstractFloat}

    nt, nq = size(m)
    nh     = length(h)

    d = zeros(T, nt, nh)
    radon_tx_forward!(d, m, dt, h, q; href = href)
    return d
end

#----------------------------------------------------------
# Adjoint parabolic Radon (t, h) → (τ, q)
#----------------------------------------------------------

"""
    radon_tx_adjoint!(m, d, dt, h, q; href = one(T))

In-place adjoint parabolic Radon transform (t–h → τ–q) matched
to `radon_tx_forward`.

Computes m_adj = Aᵗ d, where A is the forward operator.

Inputs:
  d    :: Matrix{T}          (nt × nh)   data in (t, h)
  dt   :: T                  time sampling (seconds)
  h    :: Vector{T}          offsets, length nh
  q    :: Vector{T}          Radon parameters, length nq
  href :: T (keyword)        reference offset (must be ≠ 0)

On entry:
  m    :: Matrix{T}          (nt × nq)   (will be overwritten)

On exit:
  m    contains adjoint model in (τ, q).
"""
function radon_tx_adjoint!(
    m::AbstractMatrix{T},
    d::AbstractMatrix{T},
    dt::T,
    h::AbstractVector{T},
    q::AbstractVector{T};
    href::T = one(T),
) where {T<:AbstractFloat}

    nt, nh = size(d)
    nq     = length(q)

    @assert size(m, 1) == nt
    @assert size(m, 2) == nq
    @assert href != zero(T)

    fill!(m, zero(T))

    invdt = inv(dt)

    @inbounds for ih in 1:nh
        hr  = h[ih] / href
        hr2 = hr * hr
        for iq in 1:nq
            shift = q[iq] * hr2 * invdt
            for itau in 1:nt
                # same continuous index as in forward
                x  = itau + shift
                it = floor(Int, x)

                if it >= 1 && it < nt
                    α = x - T(it)
                    # adjoint of the linear interp used in forward
                    m[itau, iq] += (one(T) - α) * d[it, ih] + α * d[it+1, ih]
                end
            end
        end
    end

    return m
end

"""
    radon_tx_adjoint(d, dt, h, q; href = one(T))

Allocating adjoint parabolic Radon transform.

Returns:
  m :: Matrix{T}  (nt × nq)  adjoint model in (τ, q)
"""
function radon_tx_adjoint(
    d::AbstractMatrix{T},
    dt::T,
    h::AbstractVector{T},
    q::AbstractVector{T};
    href::T = one(T),
) where {T<:AbstractFloat}

    nt, nh = size(d)
    nq     = length(q)

    m = zeros(T, nt, nq)
    radon_tx_adjoint!(m, d, dt, h, q; href = href)
    return m
end

#----------------------------------------------------------
# MiniOps operator: R : (τ, q) → (t, h)
#   forward: d = R * m
#   adjoint: m = R' * d
#   (no manual vec/reshape needed)
#----------------------------------------------------------

"""
    radon_tx_op(dt, h, q; href = one(T))

Return a MiniOps `Op` for the parabolic Radon transform with
linear time interpolation, operating directly on matrices.

Domain (model):  m(τ, q)  :: Matrix{T}   (nt × nq)

Range (data):    d(t, h)  :: Matrix{T}   (nt × nh)

Forward:
    d = R * m           # calls `radon_tx_forward_lin`

Adjoint:
    ma = R' * d         # calls `radon_tx_adjoint_lin`
"""
function radon_tx_op(
    dt::T,
    h::AbstractVector{T},
    q::AbstractVector{T};
    href::T = one(T),
) where {T<:AbstractFloat}

    # Size-agnostic: we let `with_shape` set (m,n) later if needed.
    f  = x -> radon_tx_forward(x, dt, h, q; href = href)
    ft = y -> radon_tx_adjoint(y, dt, h, q; href = href)

    return Op(f, ft; m = -1, n = -1, name = :radon_tx)
end


