include("naive_low_rank_approximation.jl")
include("../../functions/loss_functions/curvature_corrected_loss.jl")
include("../../functions/gradients/gradient_curvature_corrected_loss.jl")
include("../../utils/curvature_corrected_step_size.jl")

using Manifolds, Manopt

function curvature_corrected_low_rank_approximation(M, q, X, rank; stepsize=1/100, max_iter=200, change_tol=1e-6)
    n = size(X)[1]
    d = manifold_dimension(M)
    r = min(n, d, rank)

    # compute initialisation 
    R_q, U = naive_low_rank_approximation(M, q, X, r)  # ∈ T_q M^r x St(n,r)

    # compute R_q in coordinates
    Rₖₗ = reduce(hcat, get_coordinates.(Ref(M), Ref(q), R_q, Ref(DefaultOrthonormalBasis())))

    # prepare optimisation problem
    CCL(MM, V) = curvature_corrected_loss(M, q, X, U, V)
    gradCCL(MM, V) = gradient_curvature_corrected_loss(M, q, X, U, V)
    step_size = curvature_corrected_stepsize(M, q, X, U)
    println("step_size = $(step_size)")

    # do GD routine 
    ccRₖₗ = gradient_descent(Euclidean(d, r), CCL, gradCCL, Rₖₗ; stepsize=ConstantStepsize(step_size),
        stopping_criterion=StopWhenAny(StopAfterIteration(max_iter),StopWhenGradientNormLess(10.0^-8),StopWhenChangeLess(change_tol)), 
        debug=[
        :Iteration,
        (:Change, "change: %1.9f | "),
        (:Cost, " F(x): %1.11f | "),
        "\n",
        :Stop,
    ],)

    # get ccRr_q
    ccRr_q = get_vector.(Ref(M), Ref(q),[ccRₖₗ[:,l] for l=1:r], Ref(DefaultOrthonormalBasis()))
    return ccRr_q, U
end
