function stochastic_curvature_corrected_stepsize(M, q, X, U, i)
    n = size(X)
    d = manifold_dimension(M)
    r = size(U)[2]

    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    # compute Euclidean gradients
    II = CartesianIndices(n)
    A_F = 0.
    eye = Matrix(I, d, d)
    for k₁=1:d, l₁=1:r, k₂=1:d, l₂=1:r
        Aₖₗ = 0.

        ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
        Θᵢ = ONBᵢ.data.vectors
        κᵢ = ONBᵢ.data.eigenvalues

        if typeof(M) <: AbstractSphere # bug in Manifolds.jl
            κᵢ .*= distance(M, q, X[i])^2
        end
        
        Aₖₗ += sum([β(κᵢ[j])^2 * inner(M, q, get_vector(M, q, U[i,l₁] .* eye[:,k₁], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q,get_vector(M, q, U[i,l₂] .* eye[:,k₂], DefaultOrthonormalBasis()), Θᵢ[j]) for j=1:d])

        A_F += Aₖₗ^2
    end

    return 1/(2 * sqrt(A_F))
end

function curvature_corrected_stepsize(M, q, X, U::T) where {T <:Tuple}
    n = size(X)
    d = manifold_dimension(M)
    dims = length(n)
    r = Tuple([size(U[z])[2] for z=1:dims])

    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    # compute Euclidean gradients
    II = CartesianIndices(n)
    R = CartesianIndices(r)
    A_F = 0.
    eye = Matrix(I, d, d)

    for k₁=1:d, l₁ in R, k₂=1:d, l₂ in R
        Aₖₗ = 0.
        for i in II
            ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
            Θᵢ = ONBᵢ.data.vectors
            κᵢ = ONBᵢ.data.eigenvalues

            if typeof(M) <: AbstractSphere # bug in Manifolds.jl
                κᵢ .*= distance(M, q, X[i])^2
            end
            
            Aₖₗ += sum([β(κᵢ[j])^2 * prod([U[z][i[z],l₁[z]] for z=1:dims]) * prod([U[z][i[z],l₂[z]] for z=1:dims]) * inner(M, q, get_vector(M, q, eye[:,k₁], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, q,get_vector(M, q, eye[:,k₂], DefaultOrthonormalBasis()), Θᵢ[j]) for j=1:d]) 
            
        end
        A_F += Aₖₗ^2
    end

    return 1/(2 * sqrt(A_F))
end