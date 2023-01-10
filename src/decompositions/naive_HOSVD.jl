include("naive_SVD.jl")

function naive_HOSVD(M, q, X)
    D = size(X)
    @assert size(D)[1] > 1 and size(D)[1] < 4

    # TODO make list for the U's
    for i in 1:size(D)[1]
        Dᵢ = D[1:end .!= i]
        dᵢ = D[i]
        # construct power manifold and base point
        Mᵢ = PowerManifold(M, Dᵢ)
        qᵢ = fill(q, Dᵢ)
        # construct Xᵢ
        if size(D) == 2
            if i == 1
                Xᵢ = [X[k,:] for k in 1:dᵢ]
            else
                Xᵢ = [X[:,k] for k in 1:dᵢ]
            end
        elseif size(D) == 3
            if i == 1
                Xᵢ = [X[k,:,:] for k in 1:dᵢ]
            elseif i == 2
                Xᵢ = [X[:,k,:] for k in 1:dᵢ]
            else
                Xᵢ = [X[:,:,k] for k in 1:dᵢ]
            end
        end
        # compute Uᵢ from naive naive_SVD
        R_qᵢ, Uᵢ = naive_SVD(Mᵢ, qᵢ, Xᵢ)
        # stash Uᵢ TODO
    end
    # compute R_q TODO
    
end