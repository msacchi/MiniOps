#----------------------------------------------------------
# Frequency-domain linear Radon using chirp factorization
# Fast Forward and Adjoint using Chirp-transform valid for linear case with regular
# sampling in space and ray-parameter
#
# Full operator:
#
#     m(τ,p)  ->  d(t,x)
#
# Internally:
#
#     FFT in time
#     for each positive frequency ω:
#         d̂(ω,x) = R(ω) m̂(ω,p)
#     Hermitian symmetry
#     IFFT in time
#
# Adjoint:
#
#     d(t,x)  ->  m(τ,p)
#
#----------------------------------------------------------

"""
    linear_radon_chirp(v, ω, x0, dx, nx, p0, dp, np; adjoint=false)

Apply one frequency slice of the chirp-factorized linear Radon transform.
Forward mode maps `np` ray-parameter samples to `nx` spatial samples;
adjoint mode applies the exact conjugate-transpose chirp factorization.
"""
function linear_radon_chirp(
    v::AbstractVector{<:Complex},
    ω::Real,
    x0::Real,
    dx::Real,
    nx::Int,
    p0::Real,
    dp::Real,
    np::Int;
    adjoint::Bool = false,
)

    if !adjoint
        length(v) == np ||
            throw(ArgumentError("forward mode requires length(v) = np"))

        nin  = np
        nout = nx
        α = ω * dx * dp
    else
        length(v) == nx ||
            throw(ArgumentError("adjoint mode requires length(v) = nx"))

        nin  = nx
        nout = np
        α = -ω * dx * dp
    end

    nfft = 1
    while nfft < nin + nout - 1
        nfft *= 2
    end

    a = zeros(ComplexF64, nfft)

    if !adjoint
        @inbounds for m in 0:np-1
            a[m+1] =
                v[m+1] *
                exp(1im * ω * (m * dp * x0)) *
                exp(1im * α * m^2 / 2)
        end
    else
        @inbounds for n in 0:nx-1
            a[n+1] =
                v[n+1] *
                exp(-1im * ω * (p0 * n * dx)) *
                exp(1im * α * n^2 / 2)
        end
    end

    b = zeros(ComplexF64, nfft)

    @inbounds for l in -(nin-1):(nout-1)
        b[l + nin] = exp(-1im * α * l^2 / 2)
    end

    cfull = ifft(fft(a) .* fft(b))

    out = zeros(ComplexF64, nout)

    if !adjoint
        @inbounds for n in 0:nx-1
            idx = n + (nin - 1)

            out[n+1] =
                exp(1im * ω * (p0 * x0 + p0 * n * dx)) *
                exp(1im * α * n^2 / 2) *
                cfull[idx + 1]
        end
    else
        @inbounds for m in 0:np-1
            idx = m + (nin - 1)

            out[m+1] =
                exp(-1im * ω * (p0 * x0 + m * dp * x0)) *
                exp(1im * α * m^2 / 2) *
                cfull[idx + 1]
        end
    end

    return out
end


"""
    radon_tx_tp_chirp_forward(m, dt, x0, dx, nx, p0, dp, np; f1=0, f2=Inf)

Apply the frequency-domain chirp Radon forward transform from a real
`nt × np` `(τ,p)` model to a real `nt × nx` `(t,x)` gather. Frequencies
outside `[f1,f2]` Hz are zeroed.
"""
function radon_tx_tp_chirp_forward(
    m::AbstractMatrix{<:Real},
    dt::Real,
    x0::Real,
    dx::Real,
    nx::Int,
    p0::Real,
    dp::Real,
    np::Int;
    f1::Real = 0.0,
    f2::Real = Inf,
)

    nt, np_in = size(m)

    np_in == np ||
        throw(ArgumentError("model must have size nt × np"))

    M = fft(complex.(m), 1)
    D = zeros(ComplexF64, nt, nx)

    df  = 1 / (nt * dt)
    nyq = div(nt, 2) + 1

    @inbounds for k in 1:nyq
        f = (k - 1) * df

        if f < f1 || f > f2
            continue
        end

        ω = 2π * f

        D[k, :] .= linear_radon_chirp(
            vec(M[k, :]),
            ω,
            x0,
            dx,
            nx,
            p0,
            dp,
            np;
            adjoint = false,
        )
    end

    @inbounds for k in 2:nyq-1
        D[nt-k+2, :] .= conj.(D[k, :])
    end

    return real.(ifft(D, 1))
end


"""
    radon_tx_tp_chirp_adjoint(d, dt, x0, dx, nx, p0, dp, np; f1=0, f2=Inf)

Apply the adjoint of `radon_tx_tp_chirp_forward`, mapping a real
`nt × nx` gather to an `nt × np` `(τ,p)` array over the selected frequency
band.
"""
function radon_tx_tp_chirp_adjoint(
    d::AbstractMatrix{<:Real},
    dt::Real,
    x0::Real,
    dx::Real,
    nx::Int,
    p0::Real,
    dp::Real,
    np::Int;
    f1::Real = 0.0,
    f2::Real = Inf,
)

    nt, nx_in = size(d)

    nx_in == nx ||
        throw(ArgumentError("data must have size nt × nx"))

    D = fft(complex.(d), 1)
    M = zeros(ComplexF64, nt, np)

    df  = 1 / (nt * dt)
    nyq = div(nt, 2) + 1

    @inbounds for k in 1:nyq
        f = (k - 1) * df

        if f < f1 || f > f2
            continue
        end

        ω = 2π * f

        M[k, :] .= linear_radon_chirp(
            vec(D[k, :]),
            ω,
            x0,
            dx,
            nx,
            p0,
            dp,
            np;
            adjoint = true,
        )
    end

    @inbounds for k in 2:nyq-1
        M[nt-k+2, :] .= conj.(M[k, :])
    end

    return real.(ifft(M, 1))
end


"""
    radon_tx_tp_chirp_op(dt, x0, dx, nx, p0, dp, np; f1=0.0, f2=Inf)

MiniOps operator for the frequency-domain linear Radon transform.

Forward:

    d(t,x) = R * m(τ,p)

Adjoint:

    ma(τ,p) = R' * d(t,x)

The input model has size `nt × np`.

The output data has size `nt × nx`.

The time length `nt` is inferred from the input at runtime.
"""
function radon_tx_tp_chirp_op(
    dt::Real,
    x0::Real,
    dx::Real,
    nx::Int,
    p0::Real,
    dp::Real,
    np::Int;
    f1::Real = 0.0,
    f2::Real = Inf,
)

    f = m -> radon_tx_tp_chirp_forward(
        m,
        dt,
        x0,
        dx,
        nx,
        p0,
        dp,
        np;
        f1 = f1,
        f2 = f2,
    )

    ft = d -> radon_tx_tp_chirp_adjoint(
        d,
        dt,
        x0,
        dx,
        nx,
        p0,
        dp,
        np;
        f1 = f1,
        f2 = f2,
    )

    return Op(f, ft; m = -1, n = -1, name = :radon_tx_tp_chirp)
end
