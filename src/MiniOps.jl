module MiniOps

using LinearAlgebra      # Standard LA library
using FFTW               # FFTs

export Op,
       conv1d_op,
       conv1d_cols_op,
       sampling_op,
       trace_sampling_op,
       blending_op,
       scaling_op,
       diag_op,
       fft_op,
       nudft_nffts,
       nudft1_op,
       nudft2_op,
       nudft3_op,
       nudft_txy_op,
       pad_op,
       radon_tx_parab_op,
       radon_tx_tp_chirp_op,
       with_shape, 
       adjoint_test,
       linearity_test,
       opnorm_power,
       is_selfadjoint,
       soft_threshold,
       hard_threshold,
       ista,
       iht, 
       cgls,
       seismic_wavelet,
       edge_cosine_taper
    

# 1) Define Op first
include("core.jl")

# 2) Then methods that depend on Op (adjoint, composition, etc.)
include("algebra.jl")

# 3) Then concrete operators
include("conv.jl")
include("sampling.jl")
include("blending.jl")
include("scaling.jl")
include("fftops.jl")
include("nudftops.jl")
include("radon_parab.jl")
include("radon_chirp.jl")

# 4) Diagnostics using Op and the operators
include("diagnostics.jl")

# 5) Solvers
include("solvers.jl")

# 6) Extras
include("extras.jl")

end # module


