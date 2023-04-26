include("naive_low_multilinear_rank_approximation.jl")
include("../../functions/jacobi_field/beta.jl")

using Manifolds, Manopt

function curvature_corrected_low_multilinear_rank_approximation(M, q, X, rank)
    n = size(X)
    d = manifold_dimension(M)
    r = Tuple(min.(n, rank))
    dims = length(n)
    
    R_q, U  = naive_low_multilinear_rank_approximation(M, q, X, r)

    # construct linear system
    J = CartesianIndices(n)
    L = CartesianIndices(r)
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n

    # construct martrix βκB
    if dims == 2
        tensorU¹ = repeat(reshape(U[1], (n[1],1,1,r[1],1,1)), outer=(1,n[2],d,1,r[2],d))
        tensorU² = repeat(reshape(U[2], (1,n[2],1,1,r[2],1)), outer=(n[1],1,d,r[1],1,d))

        tensorΨq = repeat(reshape([get_vector(M, q, Matrix(I, d, d)[:,k], DefaultOrthonormalBasis()) for k in 1:d], (1,1,1,1,1,d)), outer=(n[1],n[2],d,r[1],r[2],1)) 

        ONB = get_basis.(Ref(M), Ref(q), DiagonalizingOrthonormalBasis.(log_q_X))
        βκΘq = [β(ONB[j₁].data.eigenvalues[j] * (typeof(M) <: AbstractSphere ? distance(M, q, X[j₁])^2 : 1.)) .* ONB[j₁].data.vectors[j] for j₁ in J, j=1:d]
        tensorβκΘq =  repeat(reshape(βκΘq, (n[1],n[2],d,1,1,1)), outer=(1,1,1,r[1],r[2],d))

        tensorβκB = tensorU¹ .* tensorU² .* inner.(Ref(M), Ref(q), tensorΨq, tensorβκΘq)

        βκB = reshape(tensorβκB, (prod(n) * d, prod(r) * d))

        # construct matrix A
        A = transpose(βκB) * βκB

        # construct vector βκBb
        tensorb = inner.(Ref(M), Ref(q), repeat(reshape(log_q_X, (n[1],n[2],1)), outer=(1,1,d)), βκΘq)
        b = reshape(tensorb, (prod(n) * d))
        βκBb = transpose(βκB) * b

        # solve linear system
        Vₖₗ = A\βκBb
        tensorVₖₗ = reshape(Vₖₗ, (r[1], r[2], d))

    else
        k=3
        # TODO: report some error
    end

    # get ccRr_q
    ccR_q = get_vector.(Ref(M), Ref(q),[tensorVₖₗ[l[1],l[2],:] for l in L], Ref(DefaultOrthonormalBasis()))
    return ccR_q, U

end
