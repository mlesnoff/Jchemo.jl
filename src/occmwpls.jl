Base.@kwdef mutable struct ParOccmwpls{Q <: Float}
    npoint::Int = 21   
    nlv::Int = 5
    K::Int = 5
    rep::Int = 10
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
    gamma::Q = .5
    scal::Symbol = :none
end 

struct Occmwpls{Q <: Float}
    fitm_emb::Vector{Any}
    nlv_emb::Vector{Int}
    fitm_occ::Vector{Occsdod}
    d::Matrix{Q}
    cutoff_d::Vector{Q}
    pwout::Vector{Q}
    cut_pwout::Q
    window::Vector{UnitRange{Int}}
    centrw::Vector{Int}
end

"""
    occmwpls(; kwargs...)
    occmwpls(X; kwargs...)
One-class classification (OCC) by moving window Pls (MWPLS).
* `X` : Training (reference class) X-data (n, p).
Keyword arguments:
* `npoint` : Total number of points in the sliding window (must be odd).
* `nlv` : Maximum nb. of latent variables (LVs) to consider in each Pls model.
* `K` : 
* `rep` : 
* `typcut` : Type of cutoffs. Possible values are: `:std`, `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:std` or `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.
* `gamma` : Proportion of scaled SD in the consensus (see function `outsdod`).
* `scal` : Symbol defining the column scaling of `X` and `Y`. Possible values are: `:none`, `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

The function implements an Occ by moving window Pls (Mwpls).


# References

# Examples
```julia
```
"""
occmwpls(; kwargs...) = JchemoModel(occmwpls, nothing, kwargs)

function occmwpls(X, Y; kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    q = nco(Y)                    
    Q = eltype(X)    
    par = recovkw(ParOccmwpls{Q}, kwargs).par
    @assert isodd(par.npoint) && par.npoint >= 1 "Argument 'npoint' must be an odd integer >= 1."
    nhwindow = Int((par.npoint - 1) / 2)  # half window
    rangetot = (nhwindow + 1):(p - nhwindow)
    centrw = collect(rangetot)
    nmod = length(rangetot)
    # Pre-allocation
    window = list(UnitRange{Int}, nmod)
    fitm_emb = list(nmod)
    nlv_emb = list(Int, nmod)
    fitm_occ = list(Occsdod, nmod)
    d = similar(X, n, nmod)
    # End
    j = 1
    @inbounds for i in rangetot  # define each window
        window[j] = (i - nhwindow):(i + nhwindow)
        vX = vcol(X, window[j])
        # Fit embedding model
        segm = segmkf(n, par.K; par.rep)
        pars = mpar(scal = [par.scal])
        model = plskern()
        res = gridcv(model, vX, Y; segm, score = rmsep, nlv = 0:par.nlv, pars).res
        if q == 1
            vscor = res.y1
        else
            s = nco(res)
            A = Matrix(res[:, (s - q + 1):end])
            vscor = sqrt.(rowsum(A).^2)
        end
        u = findall(vscor .== minimum(vscor))[1] 
        nlv_emb[j] = res.nlv[u]
        fitm_emb[j] = plskern(vX, Y; nlv = nlv_emb[j]) #, scal = par.scal)
        # End
        fitm_occ[j] = occsdod(fitm_emb[j], vX; nlv = nlv_emb[j], typcut = par.typcut, 
            cri = par.cri, alpha = par.alpha, gamma = par.gamma)
        d[:, j] = fitm_occ[j].d.d 
        j = j + 1
    end
    cutoff_d = colquant(d, 1 - par.alpha)
    pwout = rowsum(Q.(d .> cutoff_d')) / nmod
    cut_pwout = quantv(pwout, 1 - par.alpha)
    Occmwpls(fitm_emb, nlv_emb, fitm_occ, d, cutoff_d, pwout, cut_pwout,window, centrw)
end

function predict(object::Occmwpls, X)
    X = ensure_mat(X)
    Q = eltype(X)
    m = nro(X)
    nmod = length(object.window)
    d = similar(X, m, nmod)
    @inbounds for j in eachindex(object.window)
        d[:, j] = predict(object.fitm_occ[j], vcol(X, object.window[j])).d.d
    end
    pwout = rowsum(Q.(d .> object.cutoff_d')) / nmod
    pred = [if pwout[i] <= object.cut_pwout "in" else "out" end for i in eachindex(pwout)]
    pred = reshape(pred, m, 1)
    (pred = pred, d, pwout)
end


