invpseudohessian(x) = (t = typeof(x); error("Not defined for type $t", ))
momentmatrix(x) = (t = typeof(x); error("Not defined for type $t"))
resid(x) = (t = typeof(x); error("Not defined for type $t"))
bread(x) = (t = typeof(x); error("Not defined for type $t"))
leverage(x) = (t = typeof(x); error("Not defined for type $t"))
residualadjustment(k::AVarEstimator, x::Any) = (t = typeof(x); error("Not defined for type $t"))

