include("../signals/naive_low_rank_approximation.jl")

using Manifolds, Manopt

function naive_low_multilinear_rank_approximation(M, q, X, rank)
    n = size(X)
    dims = length(n)
    @assert dims == 2 and length(rank) == 2
    
    nUr = []
    r = []
    for l in 1:dims
        dₗ = n[1:end .!= l]
        nₗ = n[l]
        # construct power manifold and base point
        Mₗ = PowerManifold(M, NestedPowerRepresentation(), dₗ...)
        push!(r, min(nₗ, manifold_dimension(Mₗ), rank[l]))
        qₗ = fill(q, dₗ...)
        # construct Xₗ
        if dims == 2
            if l == 1
                Xₗ = [X[i,:] for i in 1:nₗ]
            else
                Xₗ = [X[:,i] for i in 1:nₗ]
            end 
            # else -> throw error
        end
        _, nUrₗ = naive_low_rank_approximation(Mₗ, qₗ, Xₗ, rank[l])
        push!(nUr, nUrₗ)
    end

    # compute nRr_q
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    nRr_q = [sum([log_q_X[k,l] * nUr[1][k, i] * nUr[2][l,j] for k=1:n[1], l=1:n[2]]) for i=1:r[1], j=1:r[2]]
    return nRr_q, nUr
end
