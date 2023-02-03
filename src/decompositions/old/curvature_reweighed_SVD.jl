include("../utils/curvature_reweighed_inner.jl")
include("../utils/curvature_reweighed_inner.jl")

using Manifolds

# Note that in the multi-dimensional case we pick a different inner product per dimension

function curvature_reweighed_SVD(M, q, X)
    n = size(X)[1]
    d = manifold_dimension(M)
    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    # compute Gramm matrix
    Gramm_mat = [curvature_reweighed_inner(M,q, X, log_q_X[i], log_q_X[j]) for i=1:n, j=1:n]
    # compute U and R_q
    r = min(n, d)
    (_, U) = eigen(Symmetric(Gramm_mat), n-r+1:n)
    R_q = [sum(U[:,i] .* log_q_X[:]) for i=1:r]
    return R_q, U
end
