using Manifolds

function tangent_space_SVD(M::AbstractManifold, q, log_q_X, rank)
    n = size(log_q_X)[1]
    r = min(n, manifold_dimension(M), rank)
    # compute Gramm matrix
    Gramm_mat  = Symmetric([inner(M, q, log_q_X[k], log_q_X[l]) for k=1:n, l=1:n])
    # compute U and R_q
    (_, U) = eigen(Gramm_mat, n-r+1:n)
    R_q = [sum(U[:,i] .* log_q_X[:]) for i=1:r]
    return R_q, U
end
