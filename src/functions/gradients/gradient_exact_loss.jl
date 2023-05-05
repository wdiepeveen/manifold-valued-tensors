using Manifolds
using LinearAlgebra

include("../jacobi_field/beta.jl")

function gradient_exact_loss(M::AbstractManifold, q, X, U, V)
    n = size(X)[1]
    r = size(U)[2]
    d = manifold_dimension(M)

    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    Ξ = get_vector.(Ref(M), Ref(q),[(U * transpose(V))[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))

    # construct matrix βκB
    tensorU = repeat(reshape(U, (n[1],1,1,r)), outer=(1,d,d,1))
    
    tensorΨq = repeat(reshape([get_vector(M, q, Matrix(I, d, d)[:,k], DefaultOrthonormalBasis()) for k in 1:d], (1,1,d,1)), outer=(n[1],d,1,r)) 
    
    ONB = get_basis.(Ref(M), Ref(q), DiagonalizingOrthonormalBasis.(log_q_X))
    βκΘq = [β(ONB[j₁].data.eigenvalues[j] * (typeof(M) <: AbstractSphere ? distance(M, q, X[j₁])^2 : 1.)) .* ONB[j₁].data.vectors[j] for j₁=1:n[1], j=1:d]
    tensorβκΘq =  repeat(reshape(βκΘq, (n[1],d,1,1)), outer=(1,1,d,r))
    
    tensorβκB = tensorU .* inner.(Ref(M), Ref(q), tensorΨq, tensorβκΘq)
    
    βκB = reshape(tensorβκB, (n[1] * d, r * d))

    # compute c ∈ nd
    Q = exp.(Ref(M), Ref(q), Ξ)
    tensorc = [inner(M, Q[j₁], log(M, Q[j₁], X[j₁]), parallel_transport_to(M, q, ONB[j₁].data.vectors[j], Q[j₁])) for j₁=1:n[1], j=1:d]
    c = reshape(tensorc, (n[1] * d,))
    # mat mul Bc and reshape into d x r
    Vgradient = -2 .* reshape(transpose(c) * βκB, (d, r))

    return Vgradient
end