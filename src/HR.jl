function aVar(k::K, m::StatisticalModel; demean=false, prewhiten=false, dof=0, mean = nothing, unscaled=false) where K<:HR
    ## To do: dealwith rank deficient
    mm = if K isa HR0 || K isa HR1
        GLM.momentmatrix(m)
    else
        GLM.momentmatrix(m).*sqrt.(adjustment(k, m))
    end
    aVar(k, mm; demean=false, prewhiten=false, dof=0, mean=mean, unscaled=unscaled)
end

avar(k, m) = m'm
    

adjustment(k, m::Any) = 1
