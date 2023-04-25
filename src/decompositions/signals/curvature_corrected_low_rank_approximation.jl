include("naive_low_rank_approximation.jl")
include("../../functions/jacobi_field/beta.jl")

using Manifolds, Manopt

function curvature_corrected_low_rank_approximation(M, q, X, rank)
    n = size(X)
    d = manifold_dimension(M)
    r = min(n[1], d, rank)

    # compute initialisation 
    R_q, U = naive_low_rank_approximation(M, q, X, r) 
    # construct linear system
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    
    # construct matrix βκB
    tensorU = repeat(reshape(U, (n[1],1,r,1)), outer=(1,d,1,d))
    
    tensorΨq = repeat(reshape([get_vector(M, q, Matrix(I, d, d)[:,k], DefaultOrthonormalBasis()) for k in 1:d], (1,1,1,d)), outer=(n[1],d,r,1)) 
    
    ONB = get_basis.(Ref(M), Ref(q), DiagonalizingOrthonormalBasis.(log_q_X))
    βκΘq = [β(ONB[j₁].data.eigenvalues[j] * (typeof(M) <: AbstractSphere ? distance(M, q, X[j₁])^2 : 1.)) .* ONB[j₁].data.vectors[j] for j₁=1:n[1], j=1:d]
    tensorβκΘq =  repeat(reshape(βκΘq, (n[1],d,1,1)), outer=(1,1,r,d))
    
    tensorβκB = tensorU .* inner.(Ref(M), Ref(q), tensorΨq, tensorβκΘq)
    
    βκB = reshape(tensorβκB, (n[1] * d, r * d))

    # construct matrix A
    A = transpose(βκB) * βκB

    # construct vector βκBb
    tensorb = inner.(Ref(M), Ref(q), repeat(reshape(log_q_X, (n[1],1)), outer=(1,d)), βκΘq)
    b = reshape(tensorb, (n[1] * d))
    βκBb = transpose(βκB) * b

    # solve linear system
    Vₖₗ = A\βκBb
    tensorVₖₗ = reshape(Vₖₗ, (r, d))

    # get ccRr_q
    ccR_q = get_vector.(Ref(M), Ref(q),[tensorVₖₗ[l,:] for l=1:r], Ref(DefaultOrthonormalBasis()))
    return ccR_q, U
end
