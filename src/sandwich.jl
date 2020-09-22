function sandwich(k::RobustVariance, B::Matrix, momentmatrix::Matrix; demean = false, prewhite = false, dof = 0.0)
    K, m = size(B)    
    n, m_m = size(momentmatrix)
    @assert m_m == m "number of rows of `momentmatrix` must be equal to the number of column of `B`"
    scale = n^2/(n-dof)
    A = lrvar(k, momentmatrix, demean = demean, scale = scale, prewhite = prewhite)   # df adjustment is built into vcov
    Symmetric(B*A*B)
end

