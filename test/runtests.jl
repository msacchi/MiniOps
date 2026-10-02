using Test
using Random
using LinearAlgebra
using MiniOps

function check_adjoint(A, x, y; tol=1e-10)
    Ax = A * x
    Aty = A' * y
    lhs = dot(vec(Ax), vec(y))
    rhs = dot(vec(x), vec(Aty))
    scale = max(abs(lhs), abs(rhs), eps(Float64))
    err = abs(lhs - rhs) / scale
    @test err < tol
    return err
end

@testset "MiniOps core and algebra" begin
    A = scaling_op(2.0)
    Ashape = with_shape(A, randn(7))
    @test size(Ashape) == (7, 7)

    B = Op(x -> x, y -> y; m=3, n=4, name=:B)
    C = Op(x -> x, y -> y; m=5, n=2, name=:C)
    @test_throws DimensionMismatch B * C
end

@testset "Convolution operators" begin
    Random.seed!(10)

    h = randn(5)
    A = conv1d_op(h)
    x = randn(10)
    y = randn(length(A * x))
    check_adjoint(A, x, y)

    hc = randn(ComplexF64, 4)
    Ac = conv1d_op(hc)
    xc = randn(ComplexF64, 9)
    yc = randn(ComplexF64, length(Ac * xc))
    check_adjoint(Ac, xc, yc)

    X = randn(20, 6)
    C = conv1d_cols_op(h)
    Y = randn(size(C * X))
    check_adjoint(C, X, Y)
end

@testset "Sampling: scalar/vector" begin
    Random.seed!(20)
    n = 20
    idx = [1, 5, 5, 13, 20]
    S = sampling_op(idx, n)
    x = randn(n)
    d = S * x

    @test d == x[idx]
    @test size(S) == (length(idx), n)
    check_adjoint(S, x, randn(length(idx)))

    # Duplicate index 5 must accumulate in the adjoint.
    e = ones(length(idx))
    xa = S' * e
    @test xa[5] == 2
    @test sum(xa) == length(idx)
end

@testset "Sampling: complete traces in 2D--5D" begin
    Random.seed!(30)

    for full_size in (
        (11, 7),
        (11, 4, 5),
        (11, 3, 4, 2),
        (11, 2, 3, 2, 2),
    )
        nt = full_size[1]
        ntr = prod(full_size[2:end])
        idx = unique([1, min(2, ntr), max(1, div(ntr, 2)), ntr])
        X = randn(full_size...)

        S = sampling_op(idx, full_size)
        St = trace_sampling_op(idx, full_size)
        Dref = reshape(X, nt, ntr)[:, idx]

        @test S * X == Dref
        @test St * X == Dref
        @test size(S * X) == (nt, length(idx))
        @test size(S' * (S * X)) == full_size
        @test size(S) == (nt * length(idx), prod(full_size))

        Y = randn(nt, length(idx))
        check_adjoint(S, X, Y)
        check_adjoint(St, X, Y)
    end
end

@testset "Sampling: repeated whole-trace indices" begin
    full_size = (8, 3, 2)
    idx = [2, 2, 5]
    S = trace_sampling_op(idx, full_size)
    D = ones(8, 3)
    Xadj = S' * D
    Xmat = reshape(Xadj, 8, 6)

    @test all(Xmat[:, 2] .== 2)
    @test all(Xmat[:, 5] .== 1)
    @test sum(Xmat) == 8 * 3
end

@testset "Scaling and diagonal operators" begin
    Random.seed!(40)

    α = 2.0 + 3.0im
    S = scaling_op(α)
    x = randn(ComplexF64, 30)
    y = randn(ComplexF64, 30)
    check_adjoint(S, x, y)

    w = randn(ComplexF64, 30)
    D = diag_op(w)
    check_adjoint(D, x, y)
end

@testset "FFT operator" begin
    Random.seed!(50)
    F = fft_op()

    x = randn(ComplexF64, 64)
    y = randn(ComplexF64, 64)
    check_adjoint(F, x, y)

    X = randn(ComplexF64, 16, 16)
    Y = randn(ComplexF64, 16, 16)
    check_adjoint(F, X, Y)
end

@testset "Operator composition" begin
    Random.seed!(60)
    h = randn(5)
    A = conv1d_op(h)
    G = scaling_op(0.3)
    F = fft_op()

    x = randn(20)
    nf = length(F * (G * (A * x)))
    idx = findall(rand(nf) .> 0.5)
    R = sampling_op(idx, nf)
    L = R * F * G * A

    y = randn(ComplexF64, length(L * x))
    check_adjoint(L, x, y)
end

@testset "Blending operator" begin
    Random.seed!(70)
    nt = 128
    ns = 7
    dt = 0.004
    source_times = [0.000, 0.137, 0.291, 0.456, 0.613, 0.774, 0.941]
    B = blending_op(source_times, dt, nt)

    D = randn(nt, ns)
    btest = randn(size(B, 1))
    check_adjoint(B, D, btest)
end
