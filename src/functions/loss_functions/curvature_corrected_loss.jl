include("../jacobi_field/beta.jl")

function curvature_corrected_loss(M::AbstractManifold, q, X, U, V)
    n = size(X)[1]
    d = manifold_dimension(M)
    # compute Ξ
    Ξ = get_vector.(Ref(M), Ref(q),[(U * transpose(V))[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))

    return curvature_corrected_loss(M, q, X, Ξ)
end

function curvature_corrected_loss(M::AbstractManifold, q, X, U::T, V) where {T <:Tuple{Matrix,Matrix}}
    n = size(X)
    r = size(V)[2:end]

    # compute Ξ
    Ξ = zero_vector(PowerManifold(M, NestedPowerRepresentation(), n...), fill(q, n...)) # ∈ T_q M^n

    I = CartesianIndices(n)
    L = CartesianIndices(r)
    for i in I
        Ξ[i] = get_vector(M, q, sum([(U[1][i[1], l[1]] * U[2][i[2], l[2]]) .* V[:, l[1], l[2]] for l in L]), DefaultOrthonormalBasis())
    end

    return curvature_corrected_loss(M, q, X, Ξ)
end

function curvature_corrected_loss(M::AbstractManifold, q, X, Ξ)
    n = size(X)
    d = manifold_dimension(M)
    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)

    # compute directions
    I = CartesianIndices(n)
    loss = 0.
    for i in I
        ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
        Θᵢ = ONBᵢ.data.vectors
        κᵢ = ONBᵢ.data.eigenvalues
        
        if typeof(M) <: AbstractSphere # bug in Manifolds.jl
            κᵢ .*= distance(M, q, X[i])^2
        end
        
        # compute loss
        loss += sum([β(κᵢ[j])^2 * inner(M, q, Ξ[i] - log_q_X[i], Θᵢ[j])^2 for j=1:d])
    end
    return loss/ref_distance
end
