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

pcasvd(; kwargs...) = JchemoModel(pcasvd, nothing, kwargs)
pcaeigen(; kwargs...) = JchemoModel(pcaeigen, nothing, kwargs)
pcaeigenk(; kwargs...) = JchemoModel(pcaeigenk, nothing, kwargs)
pcanipals(; kwargs...) = JchemoModel(pcanipals, nothing, kwargs)
pcanipalsmiss(; kwargs...) = JchemoModel(pcanipalsmiss, nothing, kwargs)
pcasph(; kwargs...) = JchemoModel(pcasph, nothing, kwargs)
pcaout(; kwargs...) = JchemoModel(pcaout, nothing, kwargs)
pcapp(; kwargs...) = JchemoModel(pcapp, nothing, kwargs)
spca(; kwargs...) = JchemoModel(spca, nothing, kwargs)

plskern(; kwargs...) = JchemoModel(plskern, nothing, kwargs)
plsnipals(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)
plsrosa(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)
plssimp(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)
plswold(; kwargs...) = JchemoModel(plsnipals, nothing, kwargs)









