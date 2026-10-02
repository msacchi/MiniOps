# ============================================================
# Utilities
# ============================================================

"""
    nudft_nffts(dim, order)

Number of FFTs required by the Taylor NUDFT approximation
in `dim` dimensions with total Taylor order `order`.
"""
function  nudft_nffts(dim::Int, order::Int) 

 return  binomial(order + dim, dim)

end 

"""
    signed_wavenumbers(n, dx)

Return FFT-ordered angular wavenumbers in rad/unit:

    [0, 1, ..., positive frequencies, negative frequencies]

with physical spacing `dx`.

For an even number of samples, the Nyquist component is
represented as the negative Nyquist wavenumber.
"""
function signed_wavenumbers(n::Int, dx::Real)

    q = vcat(
        0:fld(n - 1, 2),
        -fld(n, 2):-1
    )

    return (2π / (n * dx)) .* collect(q)
end


"""
    ufft(x)

Unitary N-dimensional FFT.
"""
ufft(x) = fft(x) / sqrt(length(x))


"""
    uifft(x)

Adjoint of `ufft`, i.e. unitary N-dimensional inverse FFT.
"""
uifft(x) = ifft(x) * sqrt(length(x))


# ============================================================
# Internal application routines
# ============================================================

"""
    taylor_forward(x, dims, terms)

Apply the Taylor approximation of the irregular Fourier
synthesis operator.

Each element of `terms` is `(s, w)` and represents

    diag(s) * U' * diag(w)

where U is the unitary FFT.
"""
function taylor_forward(x, dims, terms)

    X = reshape(x, dims...)

    T = promote_type(eltype(X), ComplexF64)
    y = zeros(T, dims...)

    for (s, w) in terms
        y .+= s .* uifft(w .* X)
    end

    return vec(y)
end


"""
    taylor_adjoint(d, dims, terms)

Apply the exact Hermitian adjoint of `taylor_forward`.

If

    A = sum_j diag(s_j) U' diag(w_j),

then

    A' = sum_j diag(conj(w_j)) U diag(conj(s_j)).
"""
function taylor_adjoint(d, dims, terms)

    D = reshape(d, dims...)

    T = promote_type(eltype(D), ComplexF64)
    x = zeros(T, dims...)

    for (s, w) in terms
        x .+= conj.(w) .* ufft(conj.(s) .* D)
    end

    return vec(x)
end



"""
    nudft1_op(δx, dx; order=3)

Construct a 1D FFT-based Taylor approximation to the
nonuniform DFT.

The irregular sample locations are

    x̃[n] = x[n] + δx[n]

where the underlying regular grid spacing is `dx`.

The operator maps Fourier coefficients to samples at the
perturbed positions.

The adjoint is the exact Hermitian adjoint of the truncated
Taylor operator.
"""
function nudft1_op(δx, dx; order::Int = 3)

    n = length(δx)
    dims = (n,)

    kx = signed_wavenumbers(n, dx)

    terms = []

    for p = 0:order

        s = δx .^ p

        w = (im .* kx) .^ p ./ factorial(p)

        push!(terms, (s, w))
    end

    f = x -> taylor_forward(x, dims, terms)

    ft = d -> taylor_adjoint(d, dims, terms)

    return Op(
        f,
        ft;
        m = n,
        n = n,
        name = :Taylor_NUDFT_1D
    )
end



"""
    nudft2_op(δx, δy, dx, dy; order=3)

Construct a 2D FFT-based Taylor approximation to the
nonuniform DFT.

The irregular coordinates are

    x̃[i,j] = x[i] + δx[i,j]
    ỹ[i,j] = y[j] + δy[i,j]

The multidimensional Taylor series is truncated by total
degree:

    p + q <= order.
"""
function nudft2_op(
    δx,
    δy,
    dx,
    dy;
    order::Int = 3
)

    size(δx) == size(δy) ||
        error("δx and δy must have the same size.")

    nx, ny = size(δx)

    dims = (nx, ny)

    kx = reshape(
        signed_wavenumbers(nx, dx),
        nx, 1
    )

    ky = reshape(
        signed_wavenumbers(ny, dy),
        1, ny
    )

    terms = []

    for p = 0:order
        for q = 0:(order - p)

            s =
                (δx .^ p) .*
                (δy .^ q)

            w =
                ((im .* kx) .^ p) .*
                ((im .* ky) .^ q) ./
                (factorial(p) * factorial(q))

            push!(terms, (s, w))
        end
    end

    f = x -> taylor_forward(x, dims, terms)

    ft = d -> taylor_adjoint(d, dims, terms)

    N = nx * ny

    return Op(
        f,
        ft;
        m = N,
        n = N,
        name = :Taylor_NUDFT_2D
    )
end


"""
    nudft3_op(δx, δy, δz, dx, dy, dz; order=3)

Construct a 3D FFT-based Taylor approximation to the
nonuniform DFT.

The Taylor expansion is truncated by total degree

    p + q + r <= order.
"""
function nudft3_op(
    δx,
    δy,
    δz,
    dx,
    dy,
    dz;
    order::Int = 3
)

    size(δx) == size(δy) == size(δz) ||
        error("δx, δy and δz must have the same size.")

    nx, ny, nz = size(δx)

    dims = (nx, ny, nz)

    kx = reshape(
        signed_wavenumbers(nx, dx),
        nx, 1, 1
    )

    ky = reshape(
        signed_wavenumbers(ny, dy),
        1, ny, 1
    )

    kz = reshape(
        signed_wavenumbers(nz, dz),
        1, 1, nz
    )

    terms = []

    for p = 0:order
        for q = 0:(order - p)
            for r = 0:(order - p - q)

                s =
                    (δx .^ p) .*
                    (δy .^ q) .*
                    (δz .^ r)

                w =
                    ((im .* kx) .^ p) .*
                    ((im .* ky) .^ q) .*
                    ((im .* kz) .^ r) ./
                    (
                        factorial(p) *
                        factorial(q) *
                        factorial(r)
                    )

                push!(terms, (s, w))
            end
        end
    end

    f = x -> taylor_forward(x, dims, terms)

    ft = d -> taylor_adjoint(d, dims, terms)

    N = nx * ny * nz

    return Op(
        f,
        ft;
        m = N,
        n = N,
        name = :Taylor_NUDFT_3D
    )
end


using FFTW
using LinearAlgebra

"""
    nudft_txy_op(δx, δy, nt, dx, dy; order=2)

Construct the full time-space operator

    M(ω,kx,ky)  ->  d(t,xirr,yirr)

using

    * a Taylor-NUDFT operator in (x,y),
    * a regular unitary inverse FFT in time.

The adjoint performs

    d(t,xirr,yirr) -> M(ω,kx,ky)

using

    * a regular unitary FFT in time,
    * the exact adjoint of the Taylor-NUDFT in (x,y).

Arrays are assumed to have dimensions

    (nt, nx, ny)

with time as the first dimension.
"""
function nudft_txy_op(
    δx,
    δy,
    nt::Int,
    dx,
    dy;
    order::Int = 2
)

    size(δx) == size(δy) ||
        error("δx and δy must have the same size.")

    nx, ny = size(δx)

    N = nt * nx * ny

    # Spatial operator:
    #
    #   spectrum(kx,ky) -> irregular data(x,y)
    #
    Axy = nudft2_op(
        δx,
        δy,
        dx,
        dy;
        order = order
    )

    # --------------------------------------------------------
    # Forward:
    #
    # M(ω,kx,ky)
    #      |
    #      | Axy independently for each ω
    #      v
    # D(ω,xirr,yirr)
    #      |
    #      | IFFT in time
    #      v
    # d(t,xirr,yirr)
    # --------------------------------------------------------

    function f(m)

        M = reshape(m, nt, nx, ny)

        T = promote_type(eltype(M), ComplexF64)

        Dω = zeros(T, nt, nx, ny)

        for iw = 1:nt

            mi = vec(@view M[iw, :, :])

            di = Axy * mi

            @views Dω[iw, :, :] .= reshape(di, nx, ny)
        end

        # Unitary inverse FFT in time
        d = sqrt(nt) .* ifft(Dω, 1)

        return vec(d)
    end


    # --------------------------------------------------------
    # Adjoint:
    #
    # d(t,xirr,yirr)
    #      |
    #      | FFT in time
    #      v
    # D(ω,xirr,yirr)
    #      |
    #      | Axy' independently for each ω
    #      v
    # M(ω,kx,ky)
    # --------------------------------------------------------

    function ft(d)

        D = reshape(d, nt, nx, ny)

        # Adjoint of unitary IFFT
        Dω = fft(D, 1) ./ sqrt(nt)

        T = promote_type(eltype(Dω), ComplexF64)

        M = zeros(T, nt, nx, ny)

        for iw = 1:nt

            di = vec(@view Dω[iw, :, :])

            mi = Axy' * di

            @views M[iw, :, :] .= reshape(mi, nx, ny)
        end

        return vec(M)
    end


    return Op(
        f,
        ft;
        m = N,
        n = N,
        name = :NUDFT_TXY
    )
end
