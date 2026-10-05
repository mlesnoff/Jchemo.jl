Base.@kwdef mutable struct ParMwplsr
    npoint::Int = 21
    nlv::Int = 15
    K::Int = 5
    rep::Int = 10
    h::Float = .5                    
    scal::Symbol = :none
end 

struct Mwplsr{Q <: Float}
    fitm_emb::Vector{Plsr}
    nlv_emb::Vector{Int}
    scor_emb::Vector{Q}
    d_emb::Vector{Q}
    w_emb::Vector{Q}
    vi::Vector{Q}
    window::Vector{UnitRange{Int}}
    xsel::Vector{Int}
end

"""
    mwplsr(; kwargs...)
    mwplsr(X, Y; kwargs...)
    mwplsr(X::AbstractMatrix{Q}, Y::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
Moving window Plsr (MWPLSR).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `npoint` : Total number of points in the sliding window (must be odd).
* `nlv` : Maximum nb. of latent variables (LVs) to consider in each Pls model.
* `scal` : Symbol defining the column scaling of `X` and `Y`. Possible values are: `:none`, 
    `std` (uncorrected STD), `prt` (pareto) and `:mad` (MAD).
* `K` : 
* `rep` : 
* `h` : 
* `scal` : 

# References

# Examples
```julia
```
"""
mwplsr(; kwargs...) = JchemoModel(mwplsr, nothing, kwargs)

function mwplsr(X, Y; kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = pweight(ones(eltype(X), nro(X)))
    mwplsr(X, Y, weights; kwargs...)
end

function mwplsr(X::AbstractMatrix{Q}, Y::AbstractMatrix{Q}, weights::Jchemo.ProbabilityWeights{Q}; 
    kwargs...) where Q <: Jchemo.Float

    n, p = size(X)
    q = nco(Y)                    
    par = recovkw(ParMwplsr, kwargs).par
    @assert isodd(par.npoint) && par.npoint >= 1 "Argument 'npoint' must an odd integer >= 1."
    
    nhwindow = Int((par.npoint - 1) / 2)  # half window
    rangesel = (nhwindow + 1):(p - nhwindow)
    xsel = collect(rangesel)
    nmod = length(rangesel)
    
    window = list(UnitRange{Int}, nmod)
    fitm_emb = list(Plsr, nmod)
    nlv_emb = list(Int, nmod)
    scor_emb = list(Q, nmod)
        
    j = 1   # storing index
    @inbounds for i in rangesel
    
        # Define the window
        #i = rangesel[1]
        window[j] = (i - nhwindow):(i + nhwindow)
        vX = vcol(X, window[j])
        
        # Cross-validate to select the model
        segm = segmkf(n, par.K; par.rep)
        pars = mpar(scal = [par.scal])
        model = plskern()
        res = gridcv(model, vX, Y; segm, score = rmsep, 
            nlv = 0:par.nlv, pars).res
        if q == 1
            vscor = res.y1
        else
            s = nco(res)
            A = Matrix(res[:, (s - q + 1):end])
            vscor = sqrt.(rowsum(A).^2)
        end
        u = findall(vscor .== minimum(vscor))[1] 
        res[u, :]

        # Fit the selected model        
        nlv_emb[j] = res.nlv[u]
        fitm_emb[j] = plskern(vX, Y, weights; nlv = nlv_emb[j]) #, scal = par.scal)
        scor_emb[j] = vscor[u]
                
        j = j + 1
    
    end

    # Model weights
    d_emb = scor_emb .- minimum(scor_emb)
    criw = 3. ; squared = false
    w_emb = winvs(d_emb; h = par.h, criw, squared)
    wtot = sum(w_emb)
    @. w_emb /= wtot

    # Variable importances 
    V = -ones(Q, nmod, p)
    @inbounds for i in eachindex(fitm_emb)
        V[i, window[i]] .= w_emb[i]
    end
    vi = similar(X, p)
    @inbounds for j in axes(V, 2)
        v = vcol(V, j)
        vi[j] = meanv(v[v .> -1])
    end
    
    Mwplsr(fitm_emb, nlv_emb, scor_emb, d_emb, w_emb, vi, window, xsel) 

end

function predict(object::Mwplsr, X)
    
    X = ensure_mat(X)
    Q = eltype(X)
    m = nro(X)
    q = length(object.fitm_emb[1].ymeans) 
    nmod = length(object.fitm_emb)

    pred_mod = similar(X, m, q, nmod)
    @inbounds for i in eachindex(object.fitm_emb)
        pred_mod[:, :, i] = predict(object.fitm_emb[i], vcol(X, object.window[i])).pred
    end

    pred = zeros(Q, m, q)
    @inbounds for k in 1:nmod
        pred = pred + object.w_emb[k] * pred_mod[:, :, k] 
    end

    (pred = pred, pred_mod)

end


