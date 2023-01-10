include("../utils/beta.jl")

using Manifolds

# Note that in the multi-dimensional case we pick a different inner product per dimension

function CR_SVD(M, q, X)
    n = size(X)[1]
    d = manifold_dimension(M)
    # powerM = PowerManifold(M, n)
    # powerq = fill(q, n)
    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    # compute Gramm matrix
    Gramm_mat = zeros(n,n)
    for k=1:n
        Ξₖ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[k]))
        κₖ = Ξₖ.data.eigenvalues
        βₖ = [β(κₖ[l]) for l in 1:d]
        Gramm_mat += 1/n * [
            sum(
                [   
                    βₖ[l]^2 * 
                    inner(M, q, log_q_X[i], Ξₖ.data.vectors[l]) * 
                    inner(M, q, log_q_X[j], Ξₖ.data.vectors[l]) for l=1:d
                ]
            ) 
            for i=1:n, j=1:n]
    end
    # compute U and R_q
    r = min(n, d)
    (_, U) = eigen(Symmetric(Gramm_mat), n-r+1:n)
    R_q = Symmetric.([sum(U[:,i] .* log_q_X[:]) for i=1:r])
    return R_q, U
end
