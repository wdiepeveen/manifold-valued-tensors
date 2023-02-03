include("naive_SVD.jl")
include("../utils/curvature_corrected_loss.jl")
include("../utils/gradient_curvature_corrected_loss.jl")
include("../utils/positive_numbers.jl")

using Manifolds

function curvature_corrected_low_rank_approximation(M, q, X, rank)
    n = size(X)[1]
    d = manifold_dimension(M)
    r = min(n, d, rank)
    # prepare optimisation problem
    N = Stiefel(n, r) × PowerManifold(PositiveNumbers(), NestedPowerRepresentation(), r) × Stiefel(d, r)
    # compute initialisation 
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    R_q, U = naive_SVD(M, q, X)  # ∈ T_q M^r x St(n,r)
    # compute V and Sigma
    Σ = norm.(Ref(M), Ref(q), R_q)
    ΣVtop = get_coordinates.(Ref(M), Ref(q), R_q, Ref(DefaultOrthonormalBasis()))
    Vtop = ΣVtop ./ Σ
    V = reduce(hcat,Vtop)
    # V = project(Stiefel(d, r), reduce(hcat,Vtop)) # -> not necessary
    # TODO make V into a matrix and check whether it's orthonormal 
    Ur = U[:,end-r+1:end]
    Σr = Σ[end-r+1:end]
    Vr = V[:,end-r+1:end]
    # select top r components
    # expand R_q in basis
    Xi = get_vector.(Ref(M), Ref(q),[(Ur * diagm(Σr) * transpose(Vr))[i,:] for i=1:n], Ref(DefaultOrthonormalBasis()))
    println("Xi - log_q_X = $(Xi - log_q_X)")

    # OWN gradients
    Ugrad, Σgrad, Vgrad = gradient_curvature_corrected_loss(M, q, X, Ur, Σr, Vr)
    println("Testing own gradients")
    println("Ugrad = $(Ugrad)")
    println("Σgrad = $(Σgrad)")
    println("Vgrad = $(Vgrad)")
    println("Done testing own gradients")

    # println("some testing")
    UΣVt = ProductRepr(Ur, Σr, Vr)
    # println(submanifold_component(UΣVt, 1))
    # println("some testing")
    # println((submanifold_component(UΣVt, 1) * diagm(submanifold_component(UΣVt,2)) * transpose(submanifold_component(UΣVt, 3)))[1,:])
    # println("some testing")
    # println(get_vector(M, q, (submanifold_component(UΣVt, 1) * diagm(submanifold_component(UΣVt,2)) * transpose(submanifold_component(UΣVt, 3)))[1,:], DefaultOrthonormalBasis()))
    # println("done testing")

    # construct gradient
    # !!!TODO!!! check of autograd op de individule manifolds wel werkt
    r_backend = ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend())

    CCL = [(p) -> curvature_corrected_loss(M, q, X, get_vector(M, q, (submanifold_component(p, 1) * diagm(submanifold_component(p,2)) * transpose(submanifold_component(p, 3)))[i,:], DefaultOrthonormalBasis()), i) for i=1:n] 
    println(CCL[1](UΣVt))

    # TEST U grad
    CCL_u = [(u) -> curvature_corrected_loss(M, q, X, get_vector(M, q, (u * diagm(Σr) * transpose(Vr))[i,:], DefaultOrthonormalBasis()), i) for i=1:n] 

    gradU = zeros(size(Ur))
    for i in 1:n
        gradCCL_u(p) = ManifoldDiff.gradient(Stiefel(n, r), CCL_u[i], p, r_backend)
        gradU += gradCCL_u(Ur)
    end
    
    println("Testing grad U")
    println("gradCCL_u(Ur) = $(gradU)")
    println("Done testing grad U")

    # TEST sigma -> it might go wrong when we ask for an ONB as this is a power manifold
    

    func(x) = x^2
    gradfunc(x) = ManifoldDiff.gradient(PositiveNumbers(), func, x, r_backend)
    println("gradfunc(1) = $(gradfunc([1.]))")

    # CCL_σ = [(σ) -> curvature_corrected_loss(M, q, X, get_vector(M, q, (Ur * diagm(σ) * transpose(Vr))[i,:], DefaultOrthonormalBasis()), i) for i=1:n] 
    # gradCCL_σ(p) = ManifoldDiff.gradient(PowerManifold(PositiveNumbers(), NestedPowerRepresentation(), r), CCL_σ[1], p, r_backend)
    # println("Testing grad Σ")
    # println("CCL_σ(Σr) = $(CCL_σ[1](Σr))")
    # println("gradCCL_σ(Σr) = $(gradCCL_σ(Σr))")
    # println("Done testing grad Σ")

    # gradCCL = [(N, p) -> ManifoldDiff.gradient(N, CCL[i], p, r_backend) for i=1:n]
    # println(gradCCL[1](N, UΣVt))

    # do SGD routine 
    # Urcc = stochastic_gradient_descent(N, gradCCL, ProductRepr(Ur, Σr, Vr))
    # print(Urcc)


    # compute U and R_q
    # (_, U) = eigen(Symmetric(Gramm_mat), n-r+1:n)
    # R_q = [sum(U[:,i] .* log_q_X[:]) for i=1:r]
    return R_q, U
end
