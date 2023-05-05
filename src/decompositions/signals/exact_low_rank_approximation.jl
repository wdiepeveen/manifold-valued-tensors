include("naive_low_rank_approximation.jl")
include("../../functions/loss_functions/exact_loss.jl")
include("../../functions/gradients/gradient_exact_loss.jl")

using Manifolds, Manopt

function exact_low_rank_approximation(M, q, X, rank; stepsize=1/100, max_iter=200, change_tol=1e-4, print_iterates=false)
    n = size(X)[1]
    d = manifold_dimension(M)
    r = min(n, d, rank)

    # compute initialisation 
    R_q, U = naive_low_rank_approximation(M, q, X, r)  # ∈ T_q M^r x St(n,r)

    # compute V and Sigma
    Rₖₗ = reduce(hcat, get_coordinates.(Ref(M), Ref(q), R_q, Ref(DefaultOrthonormalBasis())))

    # prepare optimisation problem
    CCL(MM, V) = exact_loss(M, q, X, U, V)
    gradCCL(MM, V) = gradient_exact_loss(M, q, X, U, V)

    # do GD routine 
    R = gradient_descent(Euclidean(d, r), CCL, gradCCL, Rₖₗ; stepsize=ConstantStepsize(stepsize),
        stopping_criterion=StopWhenAny(StopAfterIteration(max_iter),StopWhenGradientNormLess(10.0^-6),StopWhenChangeLess(stepsize * change_tol)), 
        record=:Cost, return_options=true,
        debug=(print_iterates ? [
        :Iteration,
        (:Change, "change: %1.9f | "),
        (:Cost, " F(x): %1.11f | "),
        "\n",
        :Stop,
    ] : []),)

    costs = get_record(R)
    ccRₖₗ =get_solver_result(R)
    # get ccRr_q
    ccRr_q = get_vector.(Ref(M), Ref(q),[ccRₖₗ[:,l] for l=1:r], Ref(DefaultOrthonormalBasis()))
    return (ccRr_q, U), costs
end
