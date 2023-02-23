include("naive_low_rank_approximation.jl")
include("../../functions/loss_functions/decoupled_curvature_corrected_loss.jl")
include("../../functions/gradients/decoupled_gradient_curvature_corrected_loss.jl")

using Manifolds, Manopt

function decoupled_curvature_corrected_low_rank_approximation(M, q, X, rank; stepsize=1/100, max_iter=200, change_tol=1e-7)
    n = size(X)[1]
    d = manifold_dimension(M)
    r = min(n, d, rank)

    # compute initialisation 
    R_q, U = naive_low_rank_approximation(M, q, X, r)  # ∈ T_q M^r x St(n,r)

    println(U[1,:])

    # prepare optimisation problem
    Rr = Euclidean(r)
    CCL = [(MM, p) -> curvature_corrected_loss(M, q, X[i], R_q, p) for i in 1:n]
    gradCCL = [(MM, p) -> gradient_curvature_corrected_loss(M, q, X[i], R_q, p) for i in 1:n]

    step_sizes = zeros(n)
    for i in 1:n
        log_q_Xᵢ = log(M, q, X[i])
        ONBᵢ = get_basis(M, q, DiagonalizingOrthonormalBasis(log_q_Xᵢ))
        Θᵢ = ONBᵢ.data.vectors
        κᵢ = ONBᵢ.data.eigenvalues
        
        # compute loss
        step_sizes[i] = 1 / (2 * sqrt( sum([sum([β(κᵢ[j])^2 * inner(M, q, R_q[k], Θᵢ[j]) *inner(M, q, R_q[l], Θᵢ[j]) for j=1:d])^2 for k in 1:r, l in 1:r])))
    end
    println(step_sizes)

    # do GD routine 
    ccU = [gradient_descent(Rr, CCL[i], gradCCL[i], U[i,:]; stepsize=ConstantStepsize(step_sizes[i]/2),
        stopping_criterion=StopWhenAny(StopAfterIteration(max_iter),StopWhenGradientNormLess(10.0^-8),StopWhenChangeLess(change_tol)), 
        debug=[
        :Iteration,
        (:Change, "change: %1.9f | "),
        (:Cost, " F(x): %1.11f | "),
        "\n",
        :Stop,
    ],) for i in 1:n]
    println("finshed solver")
    for i in 1:n
        U[i,:] = ccU[i]
    end
    println(U[1,:])

    return R_q, U
end
