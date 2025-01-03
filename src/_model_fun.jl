Base.@kwdef mutable struct JchemoModel{T <: Function, K <: Base.Pairs}
    algo::T   
    fitm
    kwargs::K
end

detrend_lo(; kwargs...) = JchemoModel(detrend_lo, nothing, kwargs)
detrend_pol(; kwargs...) = JchemoModel(detrend_pol, nothing, kwargs)
detrend_asls(; kwargs...) = JchemoModel(detrend_asls, nothing, kwargs)
detrend_arpls(; kwargs...) = JchemoModel(detrend_arpls, nothing, kwargs)
detrend_airpls(; kwargs...) = JchemoModel(detrend_airpls, nothing, kwargs)
fdif(; kwargs...) = JchemoModel(fdif, nothing, kwargs)
interpl(; kwargs...) = JchemoModel(interpl, nothing, kwargs)
mavg(; kwargs...) = JchemoModel(mavg, nothing, kwargs)
savgol(; kwargs...) = JchemoModel(savgol, nothing, kwargs)
snv(; kwargs...) = JchemoModel(snv, nothing, kwargs)



