include("../../functions/loss_functions/curvature_corrected_loss.jl")
include("../signals/curvature_corrected_low_rank_approximation.jl")

using Manifolds, Manopt

function curvature_corrected_low_multilinear_rank_approximation(M, q, X, rank; stepsize=1/100, max_iter=200, change_tol=1e-6)
    n = size(X)
    dims = length(n)
    @assert dims == 2 and length(rank) == 2
    
    ccUr = []
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
        _, ccUrₗ = curvature_corrected_low_rank_approximation(Mₗ, qₗ, Xₗ, rank[l]; stepsize=stepsize, max_iter=max_iter, change_tol=change_tol)
        push!(ccUr, ccUrₗ)
    end

    # compute ccRr_q
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ccRr_q = [sum([log_q_X[k,l] * ccUr[1][k, i] * ccUr[2][l,j] for k=1:n[1], l=1:n[2]]) for i=1:r[1], j=1:r[2]]
    return ccRr_q, ccUr
end
