using Manifolds
using LinearAlgebra
using LoopVectorization # -> if we want to do this, we need to unwrap all for loops 

include("../jacobi_field/beta.jl")

function gradient_curvature_corrected_loss(M::AbstractManifold, q, X, U, Σ, V)
    n = size(X)[1]
    d = manifold_dimension(M)

    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)
    Ξ = get_vector.(Ref(M), Ref(q),[(U .* Σ * V')[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))
    # compute Euclidean gradients
    Ugradient = zeros(size(U))
    Σgradient = 0.
    Vgradient = zeros(size(V))
    for i in 1:n
        ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
        Θᵢ = ONBᵢ.data.vectors
        κᵢ = ONBᵢ.data.eigenvalues
        
        Ugradient[i] = 2 .* sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, Σ .* V, DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) 
        Σgradient += 2 .* sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, U[i] .* V, DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d])
        Vgradient += 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, ((U[i] * Σ) .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for k=1:d]
    end

    # compute Riemannian gradients
    Ugrad = project(Sphere(n-1), U, Ugradient) ./ ref_distance
    Σgrad = Σgradient / ref_distance # we only use the smooth manifold structure, not the Riemannian structure
    Vgrad = project(Sphere(d-1), V, Vgradient) ./ ref_distance

    return ProductRepr(Ugrad, [Σgrad], Vgrad)
end

function gradient_curvature_corrected_loss(M::AbstractPowerManifold, q, X, U, Σ, V)
    n = size(X)[1]
    r = size(Σ)[1]
    d = manifold_dimension(M.manifold)
    D = manifold_dimension(M)

    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))

    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)
    Ξ = get_vector.(Ref(M), Ref(q),[(U * diagm(Σ) * transpose(V))[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))
    # compute Euclidean gradients
    Ugradient = zeros(size(U))
    Σgradient = zeros(size(Σ))
    Vgradient = zeros(size(V))
    for i in 1:n
        for k in R 
            ONBᵢₖ = get_basis(M.manifold, q[k], DiagonalizingOrthonormalBasis(log_q_X[i][k]))
            Θᵢₖ = ONBᵢₖ.data.vectors
            κᵢₖ = ONBᵢₖ.data.eigenvalues
            
            Ugradient[i,:] = 2 .* [sum([β(κᵢₖ[j])^2 * inner(M.manifold, q[k], get_vector(M, q, Σ[l] .* V[:,l], DefaultOrthonormalBasis())[k], Θᵢₖ[j]) * inner(M.manifold, q[k], Ξ[i][k] - log_q_X[i][k], Θᵢₖ[j]) for j=1:d]) for l=1:r] 
            Σgradient += 2 .* [sum([β(κᵢₖ[j])^2 * inner(M.manifold, q[k], get_vector(M, q, U[i,l] .* V[:,l], DefaultOrthonormalBasis())[k], Θᵢₖ[j]) * inner(M.manifold, q[k], Ξ[i][k] - log_q_X[i][k], Θᵢₖ[j]) for j=1:d]) for l=1:r]
            Vgradient += 2 .* [sum([β(κᵢₖ[j])^2 * inner(M.manifold, q[k], get_vector(M, q, ((U[i,l] * Σ[l]) .* Matrix(I, D, D))[:,ℓ], DefaultOrthonormalBasis())[k], Θᵢₖ[j]) * inner(M.manifold, q[k], Ξ[i][k] - log_q_X[i][k], Θᵢₖ[j]) for j=1:d]) for ℓ=1:D, l=1:r]
        end
    end

    # compute Riemannian gradients
    Ugrad = project(Stiefel(n,r), U, Ugradient) ./ ref_distance
    Σgrad = Σgradient ./ ref_distance # we only use the smooth manifold structure, not the Riemannian structure
    Vgrad = project(Stiefel(D,r),V, Vgradient) ./ ref_distance

    return ProductRepr(Ugrad, Σgrad, Vgrad)
end