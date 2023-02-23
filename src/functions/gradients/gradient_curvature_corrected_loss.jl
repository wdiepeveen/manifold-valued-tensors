using Manifolds
using LinearAlgebra
using LoopVectorization, BenchmarkTools # -> if we want to do this, we need to unwrap all for loops 

include("../jacobi_field/beta.jl")

function gradient_curvature_corrected_loss(M::AbstractManifold, q, log_q_X, βκ, Θ, U, Σ, V)
    n = size(log_q_X)[1]
    r = size(Σ)[1]
    d = manifold_dimension(M)

    # compute log
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)
    Ξ = get_vector.(Ref(M), Ref(q),[(U * diagm(Σ) * transpose(V))[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))
    # compute Euclidean gradients
    Ugradient = zeros(size(U))
    Σgradient = zeros(size(Σ))
    Vgradient = zeros(size(V))
    Ξ_log_q_X_Θ = 0.
    ek =  Matrix(I, d, d)
    tvector_tmp = zero_vector(M, q)
    # @turbo warn_check_args=false 
    @time for i in 1:n
        for j in 1:d
            Ξ_log_q_X_Θ = inner(M, q, Ξ[i] - log_q_X[i], Θ[i,j])
            for l in 1:r
                tvector_tmp = get_vector(M, q, Σ[l] .* V[:,l], DefaultOrthonormalBasis())
                Ugradient[i,l] += 2 * βκ[i,j]^2 * inner(M, q, tvector_tmp, Θ[i,j]) * Ξ_log_q_X_Θ
                
                tvector_tmp = get_vector(M, q, U[i,l] .* V[:,l], DefaultOrthonormalBasis())
                Σgradient[l] += 2 * βκ[i,j]^2 * inner(M, q, tvector_tmp, Θ[i,j]) * Ξ_log_q_X_Θ

                for k in 1:d
                    tvector_tmp = get_vector(M, q, (U[i,l] * Σ[l]) .* ek[:,k], DefaultOrthonormalBasis())
                    Vgradient[k,l] += 2 * βκ[i,j]^2 * inner(M, q, tvector_tmp, Θ[i,j]) * Ξ_log_q_X_Θ
                end
            end
        end
    end

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