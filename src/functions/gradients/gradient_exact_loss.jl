using Manifolds
using LinearAlgebra

include("../jacobi_field/beta.jl")

function gradient_exact_loss(M::AbstractManifold, q, X, U, V)
    n = size(X)[1]
    r = size(U)[2]
    d = manifold_dimension(M)

    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)
    Ξ = get_vector.(Ref(M), Ref(q),[(U * transpose(V))[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))
    # compute Euclidean gradients
    Vgradient = zeros(size(V))
    for i in 1:n
        ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_X[i]))
        Θᵢ = ONBᵢ.data.vectors
        κᵢ = ONBᵢ.data.eigenvalues
        
        if typeof(M) <: AbstractSphere # bug in Manifolds.jl
            κᵢ .*= distance(M, q, X[i])^2
        end

        # compute (exp_q (Ξᵢ))
        qᵢ = exp(M, q, Ξ[i])
        
        Vgradient += - 2 .* [sum([β(κᵢ[j]) * inner(M, q, get_vector(M, q, (U[i,l] .* Matrix(I, d, d))[:,k], DefaultOrthonormalBasis()), Θᵢ[j]) * inner(M, qᵢ, log(M, qᵢ, X[i]), parallel_transport_to(M, qᵢ, Θᵢ[j], q)) for j=1:d]) for k=1:d, l=1:r]
    end

    return Vgradient
end

# TODO write version for powermanifolds