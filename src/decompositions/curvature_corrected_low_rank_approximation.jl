include("naive_SVD.jl")
include("../utils/curvature_corrected_loss.jl")
include("../utils/gradient_curvature_corrected_loss.jl")
include("../utils/positive_numbers.jl")

using Manifolds

function curvature_corrected_low_rank_approximation(M, q, X, rank; stepsize=1/100)
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
    
    Ur = U[:,end-r+1:end]
    Σr = Σ[end-r+1:end]
    Vr = V[:,end-r+1:end]

    # OWN gradients
    # grad = gradient_curvature_corrected_loss(M, q, X, Ur, Σr, Vr)
    # println("Testing own gradients")
    # println("Ugrad = $(submanifold_component(grad, 1))")
    # println("Σgrad = $(submanifold_component(grad, 2))")
    # println("Vgrad = $(submanifold_component(grad, 3))")
    # println("Done testing own gradients")

    # prepare optimisation problem
    N = Stiefel(n, r) × PowerManifold(PositiveNumbers(), NestedPowerRepresentation(), r) × Stiefel(d, r)

    CCL(MM, p) = curvature_corrected_loss(M, q, X, submanifold_component(p, 1), submanifold_component(p, 2), submanifold_component(p, 3))
    gradCCL(MM, p) = gradient_curvature_corrected_loss(M, q, X, submanifold_component(p, 1), submanifold_component(p, 2), submanifold_component(p, 3))

    # do GD routine 
    # Urcc = stochastic_gradient_descent(N, gradCCL, ProductRepr(Ur, Σr, Vr))
    # print(Urcc)
    Ξ = gradient_descent(N, CCL, gradCCL, ProductRepr(Ur, Σr, Vr); stepsize=ConstantStepsize(stepsize),debug=[
        :Iteration,
        (:Change, "change: %1.9f | "),
        (:Cost, " F(x): %1.11f | "),
        "\n",
        :Stop,
    ],)


    # compute U and R_q
    # (_, U) = eigen(Symmetric(Gramm_mat), n-r+1:n)
    # R_q = [sum(U[:,i] .* log_q_X[:]) for i=1:r]
    return R_q, U
end
