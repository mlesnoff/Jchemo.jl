Base.@kwdef mutable struct JchemoModel{T <: Function, K <: Base.Pairs}
    algo::T   
    fitm
    kwargs::K
end

detrend_lo(; kwargs...) = JchemoModel(detrend_lo, nothing, kwargs)
detrend_pol(; kwargs...) = JchemoModel(detrend_pol, nothing, kwargs)
fdif(; kwargs...) = JchemoModel(fdif, nothing, kwargs)
interpl(; kwargs...) = JchemoModel(interpl, nothing, kwargs)
mavg(; kwargs...) = JchemoModel(mavg, nothing, kwargs)
savgol(; kwargs...) = JchemoModel(savgol, nothing, kwargs)
snv(; kwargs...) = JchemoModel(snv, nothing, kwargs)

plskern(; kwargs...) = JchemoModel(plskern, nothing, kwargs)
plsnipals(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)
plsrosa(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)
plssimp(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)
plswold(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)








