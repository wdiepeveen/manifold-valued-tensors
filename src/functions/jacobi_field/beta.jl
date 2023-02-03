function β(κ)
    (κ < 0) && return sinh(sqrt(-κ)) / ( sqrt((-κ)))
    (κ > 0) && return sin(sqrt(κ)) / (sqrt(κ))
    return 1.0 # cuvature zero.
end