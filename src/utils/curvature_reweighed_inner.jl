include("beta.jl")

function curvature_reweighed_inner(M::AbstractManifold, q, X, V₁, V₂)
    # TODO V1 and V2 are just normal vectors and we return a number
    # TODO check whether V1 and V2 are tangent vectors in T_q M
    n = size(X)[1]
    d = manifold_dimension(M)
    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    # compute reweighed inner product
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

function curvature_reweighed_inner(M::AbstractPowerManifold, q, X, V₁, V₂)
    # TODO V1 and V2 are just normal vectors and we return a number
    # TODO check whether V1 and V2 are tangent vectors in T_q M
    n = size(X)[1]
    d = manifold_dimension(M.manifold)
    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    # compute reweighed inner product
    innerV₁V₂ = 0
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    for i in R
        for k=1:n
            Ξₖ = get_basis(M.manifold, q[i], DiagonalizingOrthonormalBasis(log_q_X[k][i]))
            κₖ = Ξₖ.data.eigenvalues
            βₖ = [β(κₖ[l]) for l in 1:d]
            innerV₁V₂ += 1/n *
                sum(
                    [   
                        βₖ[l]^2 * 
                        inner(M.manifold, q[i], V₁[i], Ξₖ.data.vectors[l]) * 
                        inner(M.manifold, q[i], V₂[i], Ξₖ.data.vectors[l]) for l=1:d
                    ]
                ) 
        end
    end
    return innerV₁V₂
end