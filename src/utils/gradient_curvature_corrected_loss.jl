using Manifolds
using LinearAlgebra

include("beta.jl")

function gradient_curvature_corrected_loss(M::AbstractManifold, q, X, U, Σ, V)
    n = size(X)[1]
    r = size(Σ)[1]
    d = manifold_dimension(M)

    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)
    Ξ = get_vector.(Ref(M), Ref(q),[(U * diagm(Σ) * transpose(V))[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))
    # compute Euclidean gradients
    Ugradient = zeros(size(U))
    Σgradient = zeros(size(Σ))
    Vgradient = zeros(size(V))
    for i in 1:n
        ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
        Θᵢ = ONBᵢ.data.vectors
        κᵢ = ONBᵢ.data.eigenvalues
        
        # Test gradients -- zero curvature
        # Ugradient[i,:] = 2 .* [sum([inner(M, q, get_vector(M, q, Σ[l] .* V[:,l], DefaultOrthonormalBasis()) - log_q_X[i], Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for l=1:r] 
        # Σgradient += 2 .* [sum([inner(M, q, get_vector(M, q, U[i,l] .* V[:,l], DefaultOrthonormalBasis()) - log_q_X[i], Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for l=1:r]
        # Vgradient += 2 .* [sum([inner(M, q, get_vector(M, q, ((U[i,l] * Σ[l]) .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()) - log_q_X[i], Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for k=1:d, l=1:r]

        Ugradient[i,:] = 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, Σ[l] .* V[:,l], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for l=1:r] 
        Σgradient += 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, U[i,l] .* V[:,l], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for l=1:r]
        Vgradient += 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, ((U[i,l] * Σ[l]) .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for k=1:d, l=1:r]

        # Ugradient[i,:] = 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, Σ[l] .* V[:,l], DefaultOrthonormalBasis()) - log_q_X[i], Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for l=1:r] 
        # Σgradient += 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, U[i,l] .* V[:,l], DefaultOrthonormalBasis()) - log_q_X[i], Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for l=1:r]
        # Vgradient += 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, ((U[i,l] * Σ[l]) .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()) - log_q_X[i], Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for k=1:d, l=1:r]
    end

    # compute Riemannian gradients
    Ugrad = project(Stiefel(n,r), U, Ugradient) ./ ref_distance
    Σgrad = Σgradient ./ ref_distance # sharp not implemented in Manifolds
    # Σgrad = Σ .^2 .* Σgradient ./ n # sharp not implemented in Manifolds
    Vgrad = project(Stiefel(d,r),V, Vgradient) ./ ref_distance

    return ProductRepr(Ugrad, Σgrad, Vgrad)
end

# TODO write version for powermanifolds