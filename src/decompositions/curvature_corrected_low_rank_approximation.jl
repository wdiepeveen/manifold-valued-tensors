include("naive_SVD.jl")
include("../utils/curvature_corrected_loss.jl")

using Manifolds

function curvature_reweighed_low_rank_approximation(M, q, X, rank)
    n = size(X)[1]
    d = manifold_dimension(M)
    r = min(n, d, rank)
    # prepare optimisation problem
    N = Stiefel(n, r) × PowerManifold(Hyperbolic(1), NestedPowerRepresentation(), r) × Stiefel(d, r)
    # compute initialisation 
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    R_q, U = naive_SVD(M, q, X)  # ∈ T_q M^r x St(n,r)
    # compute V and Sigma
    Σ = norm.(Ref(M), Ref(q), R_q)
    ΣVtop = get_coordinates.(Ref(M), Ref(q), R_q, Ref(DefaultOrthonormalBasis()))
    Vtop = ΣVtop ./ Σ
    V = reduce(hcat,Vtop)
    # V = project(Stiefel(d, r), reduce(hcat,Vtop)) # -> not necessary
    # TODO make V into a matrix and check whether it's orthonormal 
    Ur = U[:,end-r+1:end]
    Σr = Σ[end-r+1:end]
    Vr = V[:,end-r+1:end]
    # select top r components
    # expand R_q in basis
    Xi = get_vector.(Ref(M), Ref(q),[(Ur * Σr * transpose(Vr))[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))
    println(Xi - log_q_X)

    # construct gradient
    r_backend = ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend())
    gradCCL(N, UΣVt) = [ManifoldDiff.gradient(N, curvature_corrected_loss(M, q, X, get_vector.(Ref(M), Ref(q),(UΣVt[1] * diagm(UΣVt[2]) * transpose(UΣVt[3]))[i,:], Ref(DefaultOrthonormalBasis())), i), p, r_backend) for i=1:n]
    # do SGD routine 
    Urcc, Σrcc, Vrcc = stochastic_gradient_descent(N, gradCCL, ProductRepr(Ur, Σr, Vr))
    print(Urcc)
    # compute U and R_q
    # (_, U) = eigen(Symmetric(Gramm_mat), n-r+1:n)
    # R_q = [sum(U[:,i] .* log_q_X[:]) for i=1:r]
    return R_q, U
end
