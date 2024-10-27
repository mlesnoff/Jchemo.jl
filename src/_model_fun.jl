Base.@kwdef mutable struct JchemoModel{T <: Function, K <: Base.Pairs}
    algo::T   
    fitm
    kwargs::K
end

plskern(; kwargs...) = JchemoModel(plskern, nothing, kwargs)
plsnipals(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)
plsrosa(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)
plssimp(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)
plswold(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)









