include("beta.jl")

function curvature_corrected_loss(M::AbstractManifold, q, X, Ξᵢ, i)
    n = size(X)[1]
    d = manifold_dimension(M)
    # compute log
    log_q_Xᵢ = log(M, q, X[i])  # ∈ T_q M

    # compute directions
    ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_Xᵢ))
    Θᵢ = ONBᵢ.data.vectors
    κᵢ = ONBᵢ.data.eigenvalues
    
    # compute loss
    losses = [β(κᵢ[j])^2 * inner(M, q, Ξᵢ - log_q_Xᵢ, Θᵢ[j])^2 for j=1:d]
    return sum(losses)
end

# TODO write version for powermanifolds