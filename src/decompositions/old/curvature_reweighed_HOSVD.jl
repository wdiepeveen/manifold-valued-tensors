include("curvature_reweighed_SVD.jl")

function curvature_reweighed_HOSVD(M, q, X)  # only works for 2D and 3D atm
    n = size(X)
    dims = length(n)
    @assert dims > 1 and dims < 4

    # make list for the U's
    d = []
    U = []
    for i in 1:dims
        dᵢ = n[1:end .!= i]
        push!(d, prod(dᵢ))
        nᵢ = n[i]
        # construct power manifold and base point
        Mᵢ = PowerManifold(M, NestedPowerRepresentation(), dᵢ...)
        qᵢ = fill(q, dᵢ...)
        # construct Xᵢ
        if dims == 2
            if i == 1
                Xᵢ = [X[k,:] for k in 1:nᵢ]
            else
                Xᵢ = [X[:,k] for k in 1:nᵢ]
            end
        elseif dims == 3
            if i == 1
                Xᵢ = [X[k,:,:] for k in 1:nᵢ]
            elseif i == 2
                Xᵢ = [X[:,k,:] for k in 1:nᵢ]
            else
                Xᵢ = [X[:,:,k] for k in 1:nᵢ]
            end
        end
        # compute Uᵢ from CR_SVD
        _, Uᵢ = curvature_reweighed_SVD(Mᵢ, qᵢ, Xᵢ)
        # append Uᵢ
        push!(U,Uᵢ)
    end
    # compute R_q TODO
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    if dims == 2
        r1 = min(n[1], d[1])
        r2 = min(n[2], d[2])
        R_q = [sum([log_q_X[k,l] * U[1][k, i] * U[2][l,j] for k=1:n[1], l=1:n[2]]) for i=1:r1, j=1:r2]
        return R_q, U
    elseif dims == 3
        r1 = min(n[1], d[1])
        r2 = min(n[2], d[2])
        r3 = min(n[3], d[3])
        R_q = [sum([log_q_X[k,l] * U[1][k, i] * U[2][l,j] * U[3][m, p] for k=1:n[1], l=1:n[2], m=1:n[3]]) for i=1:r1, j=1:r2, j=1:r3]
        return R_q, U
    end
end
