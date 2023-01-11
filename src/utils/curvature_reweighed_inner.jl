include("beta.jl")

function curvature_reweighed_inner(M, q, X, V₁, V₂)
    # TODO V1 and V2 are just normal vectors and we return a number
    # TODO check whether V1 and V2 are tangent vectors in T_q M
    n = size(X)[1]
    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    # if not PowerManifold
    innerV₁V₂ = 0
    for k=1:n
        Ξₖ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[k]))
        κₖ = Ξₖ.data.eigenvalues
        βₖ = [β(κₖ[l]) for l in 1:d]
        innerV₁V₂ += 1/n *
            sum(
                [   
                    βₖ[l]^2 * 
                    inner(M, q, V₁, Ξₖ.data.vectors[l]) * 
                    inner(M, q, V₂, Ξₖ.data.vectors[l]) for l=1:d
                ]
            ) 
    end
    return innerV₁V₂
end