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
    N = Sphere(n-1) × PositiveNumbers() × Sphere(d-1)
    log_q_data = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    data = X
    println("starting for loop")
    for rr in 1:r
        CCL(MM, p) = curvature_corrected_loss(M, q, data, submanifold_component(p, 1), submanifold_component(p, 2), submanifold_component(p, 3))
        gradCCL(MM, p) = gradient_curvature_corrected_loss(M, q, data, submanifold_component(p, 1), submanifold_component(p, 2), submanifold_component(p, 3))

        # do GD routine 
        @time p = gradient_descent(N, CCL, gradCCL, ProductRepr(U[:,r-rr+1], [Σ[r-rr+1]], V[:,r-rr+1]); stepsize=ConstantStepsize(stepsize),
            stopping_criterion=StopWhenAny(StopAfterIteration(max_iter),StopWhenGradientNormLess(10.0^-8),StopWhenChangeLess(change_tol)), 
            debug=[
            :Iteration,
            (:Change, "change: %1.9f | "),
            (:Cost, " F(x): %1.11f | "),
            "\n",
            :Stop,
        ],)
        
        U[:,r-rr+1] = submanifold_component(p, 1)
        Σ[r-rr+1] = submanifold_component(p, 2)[1]
        V[:,r-rr+1] = submanifold_component(p, 3)

        Ξ = get_vector.(Ref(M), Ref(q),[(U[:,r-rr+1:r] * diagm(Σ[r-rr+1:r]) * transpose(V[:,r-rr+1:r]))[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))
        println(norm.(Ref(M), Ref(q), log_q_data - Ξ))
        data = exp.(Ref(M),Ref(q), log_q_data - Ξ)
        # TODO now get rid of all components in the remaining colums of U and V that are orthogonal to what we just found
    println("finished solver")

    # TODO get ccRr_q and ccUr
    ccUr = U
    ccRr_q = get_vector.(Ref(M), Ref(q),[(diagm(Σ) * transpose(V))[i,:] for i=1:r], Ref(DefaultOrthonormalBasis()))
    return ccRr_q, ccUr
end
