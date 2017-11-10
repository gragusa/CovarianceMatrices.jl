abstract type LinModTestStat end



mutable struct AndersonRubinStat <: LinModTestStat
    stat::Float64
    pval::Float64
end


function AndersonRubinStat(l::LinearIVModel, v::RobustVariance)
end
