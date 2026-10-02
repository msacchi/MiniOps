# MiniOps.jl

A lightweight, matrix-free linear operator framework for inverse problems, signal processing, seismic imaging, and PDE/FWI kernels.

`MiniOps` allows you to write expressions such as:

```julia
y  = A * x
x̂  = A' * y
L  = R * F * A
z  = L * x
```

where `A`, `F`, `R`, … are linear **operators**, not matrices. 

The goals:

- Minimal but powerful abstraction  
- Matrix-free forward/adjoint operators 
- Shape-agnostic (1D, 2D, ND arrays)  
- Real/complex type-agnostic  
- Clean algebra: composition, adjoint, scalar multiplication  
- Essential operators:
  - 1D convolution
  - columnwise convolution
  - sampling
  - scaling & diagonal ops
  - FFT / inverse FFT
- Diagnostics: adjoint test, linearity test, norm estimate (max  eigenvalue of A'A)

MiniOps supports two design families of linear operators:

 - Size-agnostic operators (generic, work on any array)
 - Size-fixed operators (optimized, assume a known grid)

Both approaches have advantages and trade-offs. 

Size-Agnostic Operators: A size-agnostic operator does not know the shape of the input a priori.
It simply takes whatever array you give it and applies the transform.
Example: unitary FFT defined as closures:

```julia
F = fft_op()          # works for any shape
y = F * x
x2 = F' * y
```

Size-Fixed (Planned) Operators: A size-fixed operator stores shape information at construction time.
This enables precomputation and better performance.

```julia
F = fft_op(x0)        # remembers size(x0)
y = F * x
```
When to Use Which Approach

Use size-agnostic operators for:

- Classroom demonstrations.
- Quick experiments.
- Code that must accept arbitrary input.

Use size-fixed operators for:

- Production solvers.
- Iterative schemes using the solvers currently provided by MiniOps (ISTA, IHT, CGLS).
- Full waveform inversion, RTM, Radon transforms and any operator used inside a loop called many times




# 1. Installation & Usage

Inside the package folder:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Then:

```julia
using MiniOps
```

For development:

```julia
using Pkg
Pkg.develop(path="/path/to/MiniOps")
```

---

# 2. Core Concept: `Op`

An `Op` represents a matrix-free linear operator:

```julia
struct Op{F,FT}
    f::F      # forward  x -> A*x
    ft::FT    # adjoint  y -> A'*y
    m::Int    # number of elements in codomain (optional)
    n::Int    # number of elements in domain   (optional)
    name::Symbol
end
```

### Key ideas:

- `f`/`ft` are closures (forward/adjoint).
- `m` and `n` default to `-1` (shape unknown).
- Operators compose:

```julia
L = R * F * A
```

- Adjoint works automatically:

```julia
At = A'
```

- Operators work on arrays of **any shape**.

### Primary overloads:

| Expression           | Meaning                         |
|----------------------|---------------------------------|
| `A * x`              | forward                         |
| `A(x)`               | same as above                   |
| `A' * y`             | adjoint                         |
| `C = A * B`          | composition                     |
| `size(A)`            | returns `(m,n)`                 |
| `α * A`              | scalar multiplication           |

---

# 3. Helper: `with_shape`

Assign meaningful `(m,n)` based on a sample input:

```julia
A_shaped = with_shape(A, x0)
```

Sets:

- `n = length(vec(x0))`
- `m = length(vec(A*x0))`

Example:

```julia
F = fft_op()
X0 = randn(ComplexF64,128,128)
F = with_shape(F, X0)
@show size(F)
```

---

# 4. Built-in Operators

### 4.1 `conv1d_op(h)`
1D convolution operator.

```julia
A = conv1d_op(h)
y = A * x
```

---

### 4.2 `conv1d_cols_op(h)`
Columnwise convolution (applied to each trace).

```julia
A = conv1d_cols_op(h)
Y = A * X
Z = A' * Y
```

---

### 4.3 `sampling_op(idx, n)` and `trace_sampling_op(idx, full_size)`

MiniOps distinguishes scalar/vector sampling from whole-trace sampling.

For a vector:

```julia
S = sampling_op([1, 5, 10], 20)
y = S * x                    # x[[1,5,10]]
```

For seismic-style arrays with time in the first dimension, use whole-trace
sampling explicitly:

```julia
X = randn(nt, nx, ny)
S = trace_sampling_op(idx, size(X))
D = S * X                    # size (nt, length(idx))
Xa = S' * D                  # size(Xa) == size(X)
```

`sampling_op(idx, size(X))` remains backward-compatible and delegates to
whole-trace sampling whenever the size tuple has two or more dimensions.
The spatial dimensions are flattened in Julia column-major order.

---

### 4.4 `scaling_op(α)`
Scalar multiplication operator.

---

### 4.5 `diag_op(w)`
Diagonal operator: `y = w .* x`

---

### 4.6 `fft_op()`
Matrix-free FFT/bFFT operator.

```julia
F = fft_op()
Y = F * X
X2 = F' * Y
```

---

### 4.7 `pad_op()`

```julia
P = pad_op( (4,4,4), (2,2,2), (2,2,2))
A = P*randn(4,4,4)
# A has been padded with 2 samples at the beginning and end
# of each dim
```

---

# 5. Diagnostics

### 5.1 `adjoint_test(A, x, y)`
Validates inner-product identity:

$$
\langle Ax, y \rangle \approx \langle x, A'y \rangle.
$$

---

### 5.2 `linearity_test(A, x, z)`
Checks linearity.

---

### 5.3 `opnorm_power(A, x0)`
Power-method estimate of the spectral norm sn=‖A‖₂. Maximum eig of A'A is  sn²

---

### 5.4 `is_selfadjoint(A, x)`
Tests whether `A ≈ A'`.

---

# 6. Solvers

---

### 6.1 `ista(A, y, u0, mu, step_size; niter=100, verbose=false)`
Iterative Soft-Thresholding Algorithm ||A x - y||_2^2 + mu ||x||_1 

---

### 6.2 `cgls(A, b, mu, x0; tol=1e-6, max_iter=1000)`
Conjugate gradients to minimize ||A x-b||_2^2 + mu ||x||_2^2

---

# 7. Extending MiniOps

```julia
function my_op()
    f  = x -> forward_code(x)
    ft = y -> adjoint_code(y)
    Op(f, ft; m=-1, n=-1, name=:my_op)
end
```

Run:

```julia
adjoint_test(A, randn(size), randn(size))
```

---

# 8. More

Check `00_demo.ipynb`, `01_demo.ipynb`,  `02_demo.ipynb`


---




# 9. API reference

A compact LaTeX reference of the exported operators, solvers, conventions,
and examples is provided in `docs/MiniOps_API.tex`. The sampling and NUDFT
normalization conventions are described there explicitly.
