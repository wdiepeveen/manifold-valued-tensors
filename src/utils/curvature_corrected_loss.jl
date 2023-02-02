include("beta.jl")

function curvature_corrected_loss(M::AbstractManifold, q, X, U, Σ, V)
    n = size(X)[1]
    d = manifold_dimension(M)
    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)
    Ξ = get_vector.(Ref(M), Ref(q),[(U * diagm(Σ) * transpose(V))[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))

    # compute directions
    loss = 0.
    for i in 1:n
        ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
        Θᵢ = ONBᵢ.data.vectors
        κᵢ = ONBᵢ.data.eigenvalues
        
        # compute loss
        loss += sum([β(κᵢ[j])^2 * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j])^2 for j=1:d])
    end
    return loss/ref_distance
end

# TODO write version for powermanifolds