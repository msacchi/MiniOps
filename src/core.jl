"""
    Op(f, ft; m, n, name=:anonymous)

Matrix-free linear operator storing a forward action `f`, an adjoint action
`ft`, flattened codomain/domain sizes `(m, n)`, and a descriptive `name`.

Apply an operator with `A * x` or `A(x)`, obtain its adjoint with `A'`, and
compose operators with `A * B`. A size of `-1` means that the corresponding
flattened dimension is intentionally unknown until runtime.
"""
struct Op{F,FT}
    f::F
    ft::FT
    m::Int
    n::Int
    name::Symbol
end

Op(f, ft; m, n, name=:anonymous) =
    Op{typeof(f),typeof(ft)}(f, ft, m, n, name)

Base.size(A::Op) = (A.m, A.n)
(A::Op)(x) = A.f(x)
Base.:*(A::Op, x) = A.f(x)

"""
    with_shape(A::Op, x0) -> Op

Return a copy of operator `A` with flattened dimensions inferred from a
prototype input `x0`. The forward operator is applied once, so the returned
operator has `n = length(x0)` and `m = length(A * x0)`.
"""
function with_shape(A::Op, x0)
    y0 = A * x0
    m = length(y0)
    n = length(x0)
    return Op(A.f, A.ft; m=m, n=n, name=A.name)
end
