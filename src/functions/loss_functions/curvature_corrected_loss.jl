include("../jacobi_field/beta.jl")

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

function curvature_corrected_loss(M::AbstractManifold, q, X, Ξ)
    n = size(X)[1]
    d = manifold_dimension(M)
    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)

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

function curvature_corrected_loss(M::AbstractPowerManifold, q, X, U, Σ, V)
    n = size(X)[1]
    d = manifold_dimension(M)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)
    # TODO we need to construct Ξ 
    Ξ = 0. .* log_q_X
    VV = 
    for k in R
        # TODO we need to reshape V 
        Ξ[k] = get_vector.(Ref(M.manifold), Ref(q),[(U * diagm(Σ) * transpose(V))[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))
    end

    # compute directions
    loss = 0.
    for i in 1:n
        for k in R 
            ONBᵢₖ = get_basis(M.manifold, q, DiagonalizingOrthonormalBasis(log_q_X[i][k]))
            Θᵢₖ = ONBᵢₖ.data.vectorsₖ
            κᵢₖ = ONBᵢₖ.data.eigenvalues
            
            # compute loss
            loss += sum([β(κᵢₖ[j])^2 * inner(M.manifold, q, Ξ[i] - log_q_X[i], Θᵢ[j])^2 for j=1:d])
        end
    end
    return loss/ref_distance
end