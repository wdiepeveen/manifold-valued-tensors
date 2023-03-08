using Manifolds
using LinearAlgebra
using LoopVectorization # -> if we want to do this, we need to unwrap all for loops 

include("../jacobi_field/beta.jl")

function gradient_curvature_corrected_loss(M::AbstractManifold, q, X, U, V)
    n = size(X)
    d = manifold_dimension(M)
    
    r = size(U)[2]

    II = CartesianIndices(n)

    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)
    Ξ = get_vector.(Ref(M), Ref(q),[(U * transpose(V))[i,:] for i in II], Ref(DefaultOrthonormalBasis()))
    # compute Euclidean gradients
    
    Vgradient = zeros(size(V))
    for i in II
        ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
        Θᵢ = ONBᵢ.data.vectors
        κᵢ = ONBᵢ.data.eigenvalues

        if typeof(M) <: AbstractSphere # bug in Manifolds.jl
            κᵢ .*= distance(M, q, X[i])^2
        end
        
        Vgradient += 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, (U[i,l] .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j]) for j=1:d]) for k=1:d, l=1:r]
    end

    return Vgradient
end

function gradient_curvature_corrected_loss(M::AbstractManifold, q, X, U, V, i)
    n = size(X)
    d = manifold_dimension(M)
    r = size(U)[2]

    # compute log
    log_q_Xᵢ = log(M, q, X[i])  # ∈ T_q M
    Ξᵢ = get_vector(M, q, (U * transpose(V))[i,:], DefaultOrthonormalBasis())
    # compute Euclidean gradients
    
    Vgradient = zeros(size(V))

    ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_Xᵢ))
    Θᵢ = ONBᵢ.data.vectors
    κᵢ = ONBᵢ.data.eigenvalues

    if typeof(M) <: AbstractSphere # bug in Manifolds.jl
        κᵢ .*= distance(M, q, X[i])^2
    end

    innerᵢ = [inner(M, q, Ξᵢ - log_q_Xᵢ, Θᵢ[j]) for j in 1:d]

    for k=1:d, l=1:r
        Vgradient[k,l] += 2 * sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, (U[i,l] .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()), Θᵢ[j]) * innerᵢ[j] for j=1:d])
    end

    return Vgradient
end

function gradient_curvature_corrected_loss(M::AbstractManifold, q, X, U::T, V) where {T <:Tuple{Matrix,Matrix}}
    n = size(X)
    d = manifold_dimension(M)
    dims = length(n)
    r = size(V)[2:end]

    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)
    
    # compute Euclidean gradients
    Vgradient = zeros(size(V))
    II = CartesianIndices(n)
    L = CartesianIndices(r)
    for i in II 
        ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
        Θᵢ = ONBᵢ.data.vectors
        κᵢ = ONBᵢ.data.eigenvalues

        if typeof(M) <: AbstractSphere # bug in Manifolds.jl
            κᵢ .*= distance(M, q, X[i])^2
        end

        Ξᵢ = get_vector(M, q, sum([prod([U[z][i[z],l[z]] for z=1:dims]) .* V[:, l[1], l[2]] for l in L]), DefaultOrthonormalBasis())
        
        Vgradient += 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, (prod([U[z][i[z],l[z]] for z=1:dims]) .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξᵢ - log_q_X[i], Θᵢ[j]) for j=1:d]) for k=1:d, l in L]
    end

    return Vgradient
end

function gradient_curvature_corrected_loss(M::AbstractManifold, q, X, U::T, V, i) where {T <:Tuple{Matrix,Matrix}}
    n = size(X)
    d = manifold_dimension(M)
    dims = length(n)
    r = size(V)[2:end]

    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)
    
    # compute Euclidean gradients
    Vgradient = zeros(size(V))
    II = CartesianIndices(n)
    L = CartesianIndices(r)
    
    ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
    Θᵢ = ONBᵢ.data.vectors
    κᵢ = ONBᵢ.data.eigenvalues

    if typeof(M) <: AbstractSphere # bug in Manifolds.jl
        κᵢ .*= distance(M, q, X[i])^2
    end

    Ξᵢ = get_vector(M, q, sum([prod([U[z][i[z],l[z]] for z=1:dims]) .* V[:, l[1], l[2]] for l in L]), DefaultOrthonormalBasis())
    
    Vgradient += 2 .* [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, (prod([U[z][i[z],l[z]] for z=1:dims]) .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q, Ξᵢ - log_q_X[i], Θᵢ[j]) for j=1:d]) for k=1:d, l in L]
    

    return Vgradient
end
