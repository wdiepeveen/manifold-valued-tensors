include("../../functions/loss_functions/curvature_corrected_loss.jl")
include("../signals/curvature_corrected_low_rank_approximation.jl")
include("../../utils/stochastic_curvature_corrected_step_size.jl")

using Manifolds, Manopt

function stochastic_curvature_corrected_low_multilinear_rank_approximation(M, q, X, rank; max_iter=200, step_size=missing)
    n = size(X)
    d = manifold_dimension(M)
    r = Tuple(min.(n, rank))
    
    R_q, U  = naive_low_multilinear_rank_approximation(M, q, X, r)

    # compute R_q in coordinates
    Rₖₗ = zeros(d, r...)
    L = CartesianIndices(r)
    for l in L
        Rₖₗ[:, l] = get_coordinates(M, q, R_q[l], DefaultOrthonormalBasis())
    end

    # prepare optimisation problem
    J = CartesianIndices(n)
    CCL(MM, V) = curvature_corrected_loss(M, q, X, Tuple(U), V)
    gradCCL = [(MM, V) -> gradient_curvature_corrected_loss(M, q, X, Tuple(U), V, j) for j in reduce(vcat, J)]
    if ismissing(step_size)
        step_size = 10000000
        for j in J
        # step_size = curvature_corrected_stepsize(M, q, X, Tuple(U))
            step_size = min(stochastic_curvature_corrected_stepsize(M, q, X, Tuple(U), j), step_size)
        end
    end
    println(step_size)

    # do GD routine 
    ccRₖₗ = stochastic_gradient_descent(Euclidean(d, r...), gradCCL, Rₖₗ; cost=CCL, stepsize=ConstantStepsize(step_size),
        stopping_criterion=StopWhenAny(StopAfterIteration(max_iter)), #,StopWhenGradientNormLess(10.0^-8),StopWhenChangeLess(change_tol)), 
        debug=[
        :Iteration,
        (:Change, "change: %1.9f | "),
        (:Cost, " F(x): %1.11f | "),
        "\n",
        :Stop,
        500,
    ],)

    # get ccRr_q
    ccRr_q = get_vector.(Ref(M), Ref(q),[ccRₖₗ[:,l] for l in L], Ref(DefaultOrthonormalBasis()))
    return ccRr_q, U

end
