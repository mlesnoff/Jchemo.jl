"""
    occmwpca(; kwargs...)
    occmwpca(X; kwargs...)
One-class classification (OCC) by moving window Pca.
* `X` : Training X-data (n, p) on which was fitted model `fitm`.
Keyword arguments:
* `fun` : Function used to fit the Pca models (by default: `pcasvd`).
* `nlv` : Maximum nb. of latent variables (LVs) to consider i the Pca models.
* `typcut` : Type of cutoff. Possible values are: `:std`, `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:std` or `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.
* `gamma` : Proportion of scaled SD in the consensus (see function `outsdod`).
* `npoint` : Total number of points in the sliding window (must be odd).

## Examples
```julia
```
""" 
Base.@kwdef mutable struct ParOccmwpca1{Q <: Float}
    fun::Function = pcasvd
    nlv::Union{Nothing, Int} = nothing
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
    gamma::Q = .5
    npoint::Int = 11   # total nb. points of the sliding window (must be odd)
end 

struct Occmwpca1{Q <: Float}
    fitm_emb::Vector{Any}
    nlv_emb::Vector{Int}
    fitm_occ::Vector{Occsdod}
    d::Matrix{Q}
    cutoff::Vector{Q}
    pxout::Vector{Q}
    cut_pxout::Q
    rangemod::Vector{UnitRange{Int}}
    x_sel::Vector{Int}
end

function occmwpca(X; kwargs...)
    par = recovkw(ParOccmwpca1, kwargs).par
    @assert isodd(par.npoint) && par.npoint >= 1 "Argument 'npoint' must an odd integer >= 1."
    X = ensure_mat(X)
    n = nro(X)                    
    Q = eltype(X)
    nhwindow = Int((par.npoint - 1) / 2)  # half window
    range_sel = (nhwindow + 1):(p - nhwindow)
    x_sel = collect(range_sel)
    nmodel = length(range_sel)
    fitm_emb = list(nmodel)
    nlv_emb = list(Int, nmodel)
    fitm_occ = list(Occsdod, nmodel)
    d = similar(X, n, nmodel)
    rangemod = list(UnitRange{Int}, nmodel)
    j = 1
    @inbounds for i in range_sel
        #i = range_sel[1]
        rangemod[j] = (i - nhwindow):(i + nhwindow)
        #@show (i, rangemod[j])
        vX = vcol(X, rangemod[j])
        fitm_emb[j] = par.fun(vX; nlv)
        vres = summary(fitm_emb[j], vX).explvarx
        nlv_emb[j] = (1:nlv)[vres.cumpvar .> pctvar][1]
        fitm_occ[j] = occsdod(fitm_emb[j], vX; nlv = nlv_emb[j], typcut = par.typcut, 
            cri = par.cri, alpha = par.alpha, gamma = par.gamma)
        d[:, j] = fitm_occ[j].d.d 
        j = j + 1
    end
    #j = 1 ; summary(fitm_emb[j], vcol(X, rangemod[j])).explvarx
    cutoff = colquant(d, 1 - par.alpha)
    pxout = rowsum(Q.(d .> cutoff')) / nmodel
    cut_pxout = quantv(pxout, 1 - par.alpha)
    Occmwpca1(fitm_emb, nlv_emb, fitm_occ, d, cutoff, pxout, cut_pxout, rangemod, x_sel)
end

function predict(fitm::Occmwpca1, X)
    X = ensure_mat(X)
    Q = eltype(X)
    m = nro(X)
    nmodel = length(fitm.rangemod)
    d = similar(X, m, nmodel)
    @inbounds for j in eachindex(fitm.rangemod)
        d[:, j] = predict(fitm.fitm_occ[j], vcol(X, fitm.rangemod[j])).d.d
    end
    pxout = rowsum(Q.(d .> fitm.cutoff')) / nmodel
    pred = [if pxout[i] <= fitm.cut_pxout "in" else "out" end for i in eachindex(pxout)]
    pred = reshape(pred, m, 1)
    (pred = pred, d, pxout)
end


