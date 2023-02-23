include("naive_low_rank_approximation.jl")
include("../../functions/loss_functions/curvature_corrected_loss.jl")
include("../../functions/gradients/gradient_curvature_corrected_loss.jl")

using Manifolds, Manopt

function curvature_corrected_low_rank_approximation(M, q, X, rank; stepsize=1/100, max_iter=200, change_tol=1e-6)
    n = size(X)[1]
    d = manifold_dimension(M)
    r = min(n, d, rank)

    # compute initialisation 
    R_q, U = naive_low_rank_approximation(M, q, X, r)  # ∈ T_q M^r x St(n,r)

    # compute V and Sigma
    Σ = norm.(Ref(M), Ref(q), R_q)
    ΣVtop = get_coordinates.(Ref(M), Ref(q), R_q, Ref(DefaultOrthonormalBasis()))
    Vtop = ΣVtop ./ Σ
    V = reduce(hcat,Vtop)

    # prepare optimisation problem
    N = Stiefel(n, r) × PowerManifold(PositiveNumbers(), NestedPowerRepresentation(), r) × Stiefel(d, r)
    # precompute all the stuff that we can recycle
    # compute log
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)
    Ξ = get_vector.(Ref(M), Ref(q),[(U * diagm(Σ) * transpose(V))[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))
    # compute Euclidean gradients
    Ugradient = zeros(size(U))
    Σgradient = zeros(size(Σ))
    Vgradient = zeros(size(V))

    ONB = get_basis.(Ref(M), Ref(q), DiagonalizingOrthonormalBasis.(log_q_X))
    Θ = [ONB[i].data.vectors[j] for i in 1:n, j in 1:d]
    κ = [ONB[i].data.eigenvalues for i in 1:n]
    βκ = [β(κ[i][j]) for i in 1:n, j in 1:d]

    # Θ = [ONB[i].data.vectors[j] for i in 1:n, l in 1:r, j in 1:d]
    # κ = [ONB[i].data.eigenvalues for i in 1:n]
    # βs = [β.(κ[i][j]) for i in 1:n, l in 1:r, j in 1:d]
    
    # 

    CCL(MM, p) = curvature_corrected_loss(M, q, X, submanifold_component(p, 1), submanifold_component(p, 2), submanifold_component(p, 3))
    gradCCL(MM, p) = gradient_curvature_corrected_loss(M, q, X, βκ, Θ, submanifold_component(p, 1), submanifold_component(p, 2), submanifold_component(p, 3))

    # do GD routine 
    Ξ = gradient_descent(N, CCL, gradCCL, ProductRepr(U, Σ, V); stepsize=ConstantStepsize(stepsize),
        stopping_criterion=StopWhenAny(StopAfterIteration(max_iter),StopWhenGradientNormLess(10.0^-8),StopWhenChangeLess(change_tol)), 
        debug=[
        :Iteration,
        (:Change, "change: %1.9f | "),
        (:Cost, " F(x): %1.11f | "),
        "\n",
        :Stop,
    ],)

    # TODO get ccRr_q and ccUr
    ccUr = submanifold_component(Ξ, 1)
    ccRr_q = get_vector.(Ref(M), Ref(q),[(diagm(submanifold_component(Ξ, 2)) * transpose(submanifold_component(Ξ, 3)))[i,:] for i=1:r], Ref(DefaultOrthonormalBasis()))
    return ccRr_q, ccUr
end
