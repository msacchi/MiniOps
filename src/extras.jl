"""
    seismic_wavelet(; f0 = 20.0, dt = 0.002) -> w

Generate a decaying sinusoidal seismic wavelet with cosine edge tapering
and unit-energy normalization.

The wavelet is constructed as a sinusoid at frequency `f0` multiplied by
an exponential decay envelope. A cosine taper is applied at both ends to
reduce edge artifacts, and the wavelet is normalized to unit energy.

Arguments
---------
f0 : Central frequency in Hz (default = 20.0).
dt : Sampling interval in seconds (default = 0.002).

Returns
-------
w : Vector
    The generated seismic wavelet, normalized to unit energy.

Notes
-----
- The wavelet duration is proportional to 1 / f0.
- A cosine taper of length 0.2 * N is applied at both ends.
- The wavelet is normalized using the Euclidean norm.
- Useful for synthetic modeling, convolution tests, and toy seismic
  experiments.

Example
-------
w = seismic_wavelet(f0 = 30.0, dt = 0.001)
"""
function seismic_wavelet(;f0 = 20.0, dt = 0.002)
    N = floor(Int,(3.0/f0)/dt)
    w = (exp.(-6.0*collect(0:1:N)/N)).*sin.(2*pi*f0*collect(0:1:N)*dt)
    M = floor(Int,0.2*N)
    w = edge_cosine_taper(w,M);
    return w/norm(w)
end



"""
    edge_cosine_taper(h::AbstractVector, m::Int) -> h_tapered

Apply a smooth cosine-based taper to both ends of a 1D signal or wavelet.

The first `m` and last `m` samples of `h` are multiplied by a raised-cosine
window that smoothly transitions from 0 to 1 at the edges while leaving
the center portion unchanged.

Arguments
---------
h : Input vector (e.g., wavelet or 1D signal).

m : Number of samples used for tapering at each edge.

Returns
-------
h_tapered : Vector
    The tapered version of the input signal.

Notes
-----
- The total taper length is `2*m` and must not exceed the length of `h`.
- The taper uses a squared-sine profile from 0 to 1.
- The center of the signal remains unchanged.
- This is useful for suppressing edge artifacts and spectral leakage.

Example
-------
h2 = edge_cosine_taper(h, 16)
"""
function edge_cosine_taper(h::AbstractVector, m::Int)
    n = length(h)
    @assert 2m ≤ n "Taper length m too large for wavelet length"
    w = ones(Float64, n)

    t = range(0, π/2, length=m)
    left  = sin.(t).^2
    right = reverse(left)

    w[1:m] .= left
    w[end-m+1:end] .= right

    return h .* w
end

