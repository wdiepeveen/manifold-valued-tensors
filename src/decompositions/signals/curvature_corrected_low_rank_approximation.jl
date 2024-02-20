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
    ONB = get_basis.(Ref(M), Ref(q), DiagonalizingOrthonormalBasis.(log_q_X))
    if typeof(M) <: PowerManifold
        MM = M.manifold
        D = power_dimensions(M)[1]
        dd = manifold_dimension(MM)

        tensorU = repeat(reshape(U, (n[1],1,r,1,1)), outer=(1,dd,1,dd,D))

        tensorΨq = repeat(reshape([get_vector(M, q, Matrix(I, dd*D, dd*D)[:,(k-1)*dd + kk], DefaultOrthonormalBasis())[k] for kk=1:dd, k=1:D], (1, 1, 1, dd, D)), outer=(n[1], dd, r, 1,1));

        βκΘq = [β(ONB[j₁].data.bases[k].data.eigenvalues[j]) .* ONB[j₁].data.bases[k].data.vectors[j] for j₁=1:n[1], j=1:dd, k=1:D]

        tensorβκΘq = repeat(reshape(βκΘq,(n[1],dd,1,1,D)), outer=(1,1,r,dd,1));

        tensorβκB = tensorU .* inner.(Ref(MM), repeat(reshape(q,(1,1,1,1,D)),outer=(n[1],dd,r,dd,1)), tensorΨq, tensorβκΘq)  # these are all the non-zero coefficients of the tensor

        # all the curvature correction sub matrices are independent. So we can solve for each entry inividually
        βκB = [reshape(tensorβκB[:,:,:,:,k], (n[1] * dd, r * dd)) for k=1:D]

        tensorb = [inner.(Ref(MM), Ref(q[k]), repeat(reshape([log_q_X[j₁][k] for j₁=1:n[1]], (n[1],1)), outer=(1,dd)), βκΘq[:,:,k]) for k=1:D]

        b = [reshape(tensorb[k], (n[1] * dd)) for k=1:D]

        # construct matrix A
        A = [transpose(βκB[k]) * βκB[k] for k=1:D]

        # construct vector βκBb
        βκBb = [transpose(βκB[k]) * b[k] for k=1:D]

        # solve linear system
        Vₖₗ = reduce(vcat,[A[k]\βκBb[k] for k=1:D])
        tensorVₖₗ = reshape(Vₖₗ, (r, d))
    else
        tensorU = repeat(reshape(U, (n[1],1,r,1)), outer=(1,d,1,d))
    
        tensorΨq = repeat(reshape([get_vector(M, q, Matrix(I, d, d)[:,k], DefaultOrthonormalBasis()) for k in 1:d], (1,1,1,d)), outer=(n[1],d,r,1)) 
    
        βκΘq = [β(ONB[j₁].data.eigenvalues[j] * (typeof(M) <: AbstractSphere ? distance(M, q, X[j₁])^2 : 1.)) .* ONB[j₁].data.vectors[j] for j₁=1:n[1], j=1:d]

        tensorβκΘq =  repeat(reshape(βκΘq, (n[1],d,1,1)), outer=(1,1,r,d))

        tensorb = inner.(Ref(M), Ref(q), repeat(reshape(log_q_X, (n[1],1)), outer=(1,d)), βκΘq)

        tensorβκB = tensorU .* inner.(Ref(M), Ref(q), tensorΨq, tensorβκΘq)

        βκB = reshape(tensorβκB, (n[1] * d, r * d))

        b = reshape(tensorb, (n[1] * d))

        # construct matrix A
        A = transpose(βκB) * βκB

        # construct vector βκBb
        βκBb = transpose(βκB) * b

        # solve linear system
        Vₖₗ = A\βκBb
        tensorVₖₗ = reshape(Vₖₗ, (r, d))
    end

    # get ccRr_q
    ccR_q = get_vector.(Ref(M), Ref(q),[tensorVₖₗ[l,:] for l=1:r], Ref(DefaultOrthonormalBasis()))
    return ccR_q, U
end
