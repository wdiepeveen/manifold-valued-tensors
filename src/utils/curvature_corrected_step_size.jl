function curvature_corrected_stepsize(M, q, X, U)
    n = size(X)
    d = manifold_dimension(M)
    r = size(U)[2]

    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    # compute Euclidean gradients
    II = CartesianIndices(n)
    A = zeros(d, r, d, r)
    eye = Matrix(I, d, d)
    for i in II
        ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
        Θᵢ = ONBᵢ.data.vectors
        κᵢ = ONBᵢ.data.eigenvalues

        if typeof(M) <: AbstractSphere # bug in Manifolds.jl
            κᵢ .*= distance(M, q, X[i])^2
        end

        A += [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, U[i,l₁] .* eye[:,k₁], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q,get_vector(M, q, U[i,l₂] .* eye[:,k₂], DefaultOrthonormalBasis()), Θᵢ[j]) for j=1:d]) for k₁=1:d, l₁=1:r, k₂=1:d, l₂=1:r]
   
    end

    return 1/(2 * sqrt(sum(A .^2)))
end

function curvature_corrected_stepsize(M, q, X, U::T) where {T <:Tuple{Matrix,Matrix}}
# function curvature_corrected_stepsize(M, q, X, U)    
    n = size(X)
    d = manifold_dimension(M)
    dims = length(n)
    r = Tuple([size(U[z])[2] for z=1:dims])

    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    # compute Euclidean gradients
    II = CartesianIndices(n)
    R = CartesianIndices(r)
    A = zeros(d, r..., d, r...)
    eye = Matrix(I, d, d)

    for i in II
        ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
        Θᵢ = ONBᵢ.data.vectors
        κᵢ = ONBᵢ.data.eigenvalues

        if typeof(M) <: AbstractSphere # bug in Manifolds.jl
            κᵢ .*= distance(M, q, X[i])^2
        end

        A += [sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, prod([U[z][i[z],l₁[z]] for z=1:dims]) .* eye[:,k₁], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q,get_vector(M, q, prod([U[z][i[z],l₂[z]] for z=1:dims]) .* eye[:,k₂], DefaultOrthonormalBasis()), Θᵢ[j]) for j=1:d]) for k₁=1:d, l₁ in R, k₂=1:d, l₂ in R]
   
    end

    return 1/(2 * sqrt(sum(A .^2)))
end