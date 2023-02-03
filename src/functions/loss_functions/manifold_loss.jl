function manifold_loss(M::AbstractManifold, q, X, Ξ)
    log_q_X = log.(Ref(M), Ref(q), X)  # ∈ T_q M^n
    ref_distance = sum(norm.(Ref(M), Ref(q), log_q_X).^2)

    exp_q_Ξ =   exp.(Ref(M), Ref(q), Ξ)
    return sum(distance.(Ref(M), X, exp_q_Ξ).^2) / ref_distance
end