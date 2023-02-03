include("naive_SVD.jl")
include("../utils/curvature_corrected_loss.jl")
include("../utils/gradient_curvature_corrected_loss.jl")
include("../utils/positive_numbers.jl")

using Manifolds, Manopt

function curvature_corrected_low_multilinear_rank_approximation(M, q, X, rank; stepsize=1/100, max_iter=200, change_tol=1e-6) # TODO allow for rank to be an array -> we can recycle some stuff
    n = size(X)
    d = manifold_dimension(M)
    dims = length(n)
    @assert dims == 2

    # make list for the U's
    d = []
    U = []
    Σ = []
    V = []
    for l in 1:dims
        dₗ = n[1:end .!= l]
        push!(d, prod(dₗ))
        nₗ = n[l]
        # construct power manifold and base point
        Mₗ = PowerManifold(M, NestedPowerRepresentation(), dₗ...)
        qₗ = fill(q, dₗ...)
        # construct Xₗ
        if dims == 2
            if l == 1
                Xₗ = [X[i,:] for i in 1:nₗ]
            else
                Xₗ = [X[:,i] for i in 1:nₗ]
            end 
            # else -> throw error
        end
        # compute Uₗ from naive_SVD
        Rₗ_q, Uₗ = naive_SVD(Mₗ, qₗ, Xₗ)
        # append Uᵢ
        push!(U,Uₗ)
        # TODO Rₗ_q is already an n x (powersize) array -> so we can use R to iterate through it
        Σₗ = [[] for i in 1:nₗ] # not nl, but rank -> this should be further unrolled into a rank x rank matrix
        Vₗ = [[] for i in 1:nₗ] # same, but we want a rank x d vector here
        Rₗ = CartesianIndices(Tuple(dₗ...))
        for i in 1:nₗ
            Σₗ[i] =  norm.(Ref(M), Ref(q), Rₗ_q[i])
            ΣₗVₗtop = get_coordinates.(Ref(M), Ref(q), Rₗ_q[i], Ref(DefaultOrthonormalBasis()))
            Vₗtop = ΣₗVₗtop ./ Σₗ
            for k in Rₗ
                # TODO think about how we are actually working with V in the end
            end
        end

        # TODO extract Sigma and V in the old way

        # TODO I feel that i want the last dimension in V to be in an array -> if we want to multiply them easily (maybe not necessary)
        # TODO do GD

        # TODO save only the U
    end






    
    d = manifold_dimension(M)
    # r = min(n..., d, rank)

    # Assume dim = 2 
    # TODO we need to construct a powermanifold by unrulling X along both directions
    # TODO we need R to be an iterable over the powersize of tht manifold
    # TODO we need V ∈ R^{n x powersize x d} -> then we can get items through V[i, k, j]
    R = CartesianIndices(Tuple(power_size))

    # TODO split it up in n parts and pass it on to curvature_corrected_low_rank_approximation

    # compute initialisation 
    R_q, U = naive_SVD(M, q, X)  # ∈ T_q M^r x St(n,r)

    # compute V and Sigma
    Σ = norm.(Ref(M), Ref(q), R_q)
    ΣVtop = get_coordinates.(Ref(M), Ref(q), R_q, Ref(DefaultOrthonormalBasis()))
    Vtop = ΣVtop ./ Σ
    V = reduce(hcat,Vtop)

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
