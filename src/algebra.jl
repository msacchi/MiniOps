"""
    adjoint(A::Op) -> Op

Return the operator whose forward action is `A`'s stored adjoint and whose
adjoint action is `A`'s stored forward map.
"""
Base.adjoint(A::Op) = Op(A.ft, A.f; m=A.n, n=A.m, name=Symbol(A.name, "'"))

"""
    A * B

Compose two MiniOps operators. The result applies `B` first and `A` second;
its adjoint applies `A'` first and `B'` second.

When both dimensions are known, incompatible flattened dimensions are
rejected. A dimension value of `-1` is treated as unknown.
"""
function Base.:*(A::Op, B::Op)
    if A.n >= 0 && B.m >= 0 && A.n != B.m
        throw(DimensionMismatch("cannot compose $(A.name) ($(A.m), $(A.n)) with $(B.name) ($(B.m), $(B.n))"))
    end
    f = x -> A * (B * x)
    ft = y -> B' * (A' * y)
    return Op(f, ft; m=A.m, n=B.n, name=Symbol(A.name, "*", B.name))
end

"""
    c * A

Scale a MiniOps operator by scalar `c`. The adjoint correctly uses
`conj(c)` for complex scalars.
"""
function Base.:*(c::Number, A::Op)
    f = x -> c .* (A * x)
    ft = y -> conj(c) .* (A' * y)
    return Op(f, ft; m=A.m, n=A.n, name=Symbol(c, "*", A.name))
end
