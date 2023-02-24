include("../../functions/loss_functions/curvature_corrected_loss.jl")
include("../signals/curvature_corrected_low_rank_approximation.jl")

using Manifolds, Manopt

function curvature_corrected_low_multilinear_rank_approximation(M, q, X, rank; stepsize=1/100, max_iter=200, change_tol=1e-6)
    n = size(X)
    d = manifold_dimension(M)
    r = min.(n, d .* rank)
    
    R_q, U  = naive_low_multilinear_rank_approximation(M, q, X, r)
    println(size(R_q))

    # compute R_q in coordinates
    Rₖₗₘ = zeros(d, n...)
    for k in 1:d

    end

    # Rₖₗₘ = reduce(hcat, get_coordinates.(Ref(M), Ref(q), R_q, Ref(DefaultOrthonormalBasis())))

    # prepare optimisation problem
    CCL(MM, V) = curvature_corrected_loss(M, q, X, Tuple(U), V) # functions should be able to see that U is an array and V is a tensor
    # gradCCL(MM, V) = gradient_curvature_corrected_loss(M, q, X, U, V)

    println(CCL(Euclidean(d, r...), Rₖₗₘ))

    # do GD routine 
    ccRₖₗ = gradient_descent(Euclidean(d, r...), CCL, gradCCL, Rₖₗₘ; stepsize=ConstantStepsize(stepsize),
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

    # compute ccRr_q
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ccRr_q = [sum([log_q_X[k,l] * ccUr[1][k, i] * ccUr[2][l,j] for k=1:n[1], l=1:n[2]]) for i=1:r[1], j=1:r[2]]
    return ccRr_q, ccUr
end
