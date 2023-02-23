using Manifolds
using LinearAlgebra
using BenchmarkTools # -> if we want to do this, we need to unwrap all for loops 

include("../jacobi_field/beta.jl")

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

    ONB = get_basis.(Ref(M), Ref(q), DiagonalizingOrthonormalBasis.(log_q_X))
    Θ = [ONB[i].data.vectors[j] for i in 1:n, l in 1:r, j in 1:d]
    κ = [ONB[i].data.eigenvalues for i in 1:n]
    βs = [β.(κ[i][j]) for i in 1:n, l in 1:r, j in 1:d]
    Ξlog_q_X = [Ξ[i] - log_q_X[i] for i in 1:n, l in 1:r, j in 1:d]

    Ξlog_q_XΘ = inner.(Ref(M), Ref(q), Ξlog_q_X, Θ)
    
    ΣV = [get_vector(M, q, Σ[l] .* V[:,l], DefaultOrthonormalBasis()) for i in 1:n, l in 1:r, j in 1:d] # Check whether fill is faster
    Ugradient = 2 .* sum(βs.^2 .* inner.(Ref(M), Ref(q), ΣV, Θ) .* Ξlog_q_XΘ, dims=3)[:,:,1]

    UV = [get_vector(M, q, U[i,l] .* V[:,l], DefaultOrthonormalBasis()) for i in 1:n, l in 1:r, j in 1:d]
    Σgradient = 2 .* sum(βs.^2 .* inner.(Ref(M), Ref(q), UV, Θ) .* Ξlog_q_XΘ, dims=(1,3))[1,:,1]
    
    UΣ = [[get_vector(M, q, ((U[i,l] * Σ[l]) .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()) for i in 1:n, l in 1:r, j in 1:d] for k in 1:d]
    for k in 1:d
        Vgradient[k,:] = 2 .* sum(βs.^2 .* inner.(Ref(M), Ref(q), UΣ[k], Θ) .* Ξlog_q_XΘ, dims=(1,3))[1,:,1] 
    end

    # @time for i in 1:n
    #     ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
    #     Θᵢ = ONBᵢ.data.vectors
    #     κᵢ = ONBᵢ.data.eigenvalues
        
    #     # Ugradient[i,:] = 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, Σ[l] .* V[:,l], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for l=1:r] 
    #     # Σgradient += 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, U[i,l] .* V[:,l], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for l=1:r]
    #     Vgradient += 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, ((U[i,l] * Σ[l]) .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for k=1:d, l=1:r]
    # end

    # compute Riemannian gradients
    Ugrad = project(Stiefel(n,r), U, Ugradient) ./ ref_distance
    Σgrad = Σgradient ./ ref_distance # we only use the smooth manifold structure, not the Riemannian structure
    Vgrad = project(Stiefel(d,r),V, Vgradient) ./ ref_distance

    return ProductRepr(Ugrad, Σgrad, Vgrad)
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
            
            Ugradient[i,:] += 2 .* [sum([β(κᵢₖ[j])^2 * inner(M.manifold, q[k], get_vector(M, q, Σ[l] .* V[:,l], DefaultOrthonormalBasis())[k], Θᵢₖ[j]) * inner(M.manifold, q[k], Ξ[i][k] - log_q_X[i][k], Θᵢₖ[j]) for j=1:d]) for l=1:r] 
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

# 

function gradient_curvature_corrected_loss(M::AbstractManifold, q, X, U, Σ, V, i, k)
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

    ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
    Θ = [ONBᵢ.data.vectors[j] for l in 1:r, j in 1:d]
    κ = ONBᵢ.data.eigenvalues
    βs = [β.(κ[j]) for l in 1:r, j in 1:d]
    Ξlog_q_X = [Ξ[i] - log_q_X[i] for l in 1:r, j in 1:d]

    Ξlog_q_XΘ = inner.(Ref(M), Ref(q), Ξlog_q_X, Θ)
    
    ΣV = [get_vector(M, q, ((Σ[l] .* V[:,l]) .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()) for l in 1:r, j in 1:d] # Check whether fill is faster
    Ugradient[i,:] = 2 .* sum(βs.^2 .* inner.(Ref(M), Ref(q), ΣV, Θ) .* Ξlog_q_XΘ, dims=2)[:,1]
    # println(Ugradient)

    UV = [get_vector(M, q, ((U[i,l] * V[k,l]) .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()) for l in 1:r, j in 1:d]
    Σgradient = 2 .* sum(βs.^2 .* inner.(Ref(M), Ref(q), UV, Θ) .* Ξlog_q_XΘ, dims=(1,3))[1,:,1]
    # println(Σgradient)
    
    UΣ = [get_vector(M, q, ((U[i,l] * Σ[l]) .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()) for l in 1:r, j in 1:d]
    Vgradient[k,:] = 2 .* sum(βs.^2 .* inner.(Ref(M), Ref(q), UΣ, Θ) .* Ξlog_q_XΘ, dims=2)[:,1]
    # println(Vgradient)

    # ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
    # Θᵢ = ONBᵢ.data.vectors
    # κᵢ = ONBᵢ.data.eigenvalues
    
    # Ugradient[i,:] = 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, Σ[l] .* V[:,l], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for l=1:r] 
    # Σgradient += 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, U[i,l] .* V[:,l], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for l=1:r]
    # Vgradient[k,:] = 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, ((U[i,l] * Σ[l]) .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for k=1:d, l=1:r]

    # compute Riemannian gradients
    Ugrad = project(Stiefel(n,r), U, Ugradient) ./ ref_distance
    Σgrad = Σgradient ./ ref_distance # we only use the smooth manifold structure, not the Riemannian structure
    Vgrad = project(Stiefel(d,r),V, Vgradient) ./ ref_distance

    return ProductRepr(Ugrad, Σgrad, Vgrad)
end