using Manifolds

function naive_SVD(M, q, X)
    n = size(X)[1]
    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    # compute Gramm matrix
    Gramm_mat  = Symmetric([inner(M, q, log_q_X[k], log_q_X[l]) for k=1:n, l=1:n])
    # compute U and R_q
    r = min(n, manifold_dimension(M))
    (_, U) = eigen(Gramm_mat, n-r+1:n)
    println(size(U))
    println(size(log_q_X))
    R_q = [sum(U[:,i] .* log_q_X[:]) for i=1:r]
    return R_q, U
end
