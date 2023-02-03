include("naive_SVD.jl")
include("../utils/curvature_corrected_loss.jl")
include("../utils/gradient_curvature_corrected_loss.jl")
include("../utils/positive_numbers.jl")

using Manifolds, Manopt

function curvature_corrected_low_rank_approximation(M, q, X, rank; stepsize=1/100, max_iter=200, change_tol=1e-6)
    n = size(X)[1]
    d = manifold_dimension(M)
    r = min(n, d, rank)

    # compute initialisation 
    R_q, U = naive_SVD(M, q, X)  # ∈ T_q M^r x St(n,r)

    # compute V and Sigma
    Σ = norm.(Ref(M), Ref(q), R_q)
    ΣVtop = get_coordinates.(Ref(M), Ref(q), R_q, Ref(DefaultOrthonormalBasis()))
    Vtop = ΣVtop ./ Σ
    V = reduce(hcat,Vtop)

    # TODO check whether we can get R_q back from these

    # retain only top r components
    Ur = U[:,end-r+1:end]
    Σr = Σ[end-r+1:end]
    Vr = V[:,end-r+1:end]

    # prepare optimisation problem
    N = Stiefel(n, r) × PowerManifold(PositiveNumbers(), NestedPowerRepresentation(), r) × Stiefel(d, r)
    CCL(MM, p) = curvature_corrected_loss(M, q, X, submanifold_component(p, 1), submanifold_component(p, 2), submanifold_component(p, 3))
    gradCCL(MM, p) = gradient_curvature_corrected_loss(M, q, X, submanifold_component(p, 1), submanifold_component(p, 2), submanifold_component(p, 3))
    
    # do GD routine 
    Ξ = gradient_descent(N, CCL, gradCCL, ProductRepr(Ur, Σr, Vr); stepsize=ConstantStepsize(stepsize),
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
