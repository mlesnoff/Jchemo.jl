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
* `Kcv` : Number of folds in the K-fold cross validation used to select the nb. LVs for each window. 
* `repcv` : Number of repetitions of the K-fold cross validation used to select the nb. LVs for each window.
* `h` : A scalar defining the shape of the weight function computed by function `winvs`. Lower is h, 
    sharper is the function. See function `winvs` for details. 
* `scal` : Symbol defining the column scaling of `X` and `Y`. Possible values are: `:none` (default), `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

The function implements a moving (i.e., sliding) window Pls regression (Mwpls), as follows.

1) The full range of the X-columns is divided into sliding windows of `npoint` columns. The center of each 
    window is moved of one column from the previous window. Assume that the total number of windows is 'k'. 

2) For each window:

    * a Plsr model with a maximal number of `nlv` LVs (principal components) is fitted.
    
    * A repeated K-fold cross validation is run between 0:`nlv` on the training observations, and the model 
        showing the minimum average RMSEP-CV is selected and stored. 
    
    * The new observations are predicted (for the considered window) from the selected model re-fitted on all 
        the training observations. The predictions are stored.
         
3) A performance discrepancy is computed for each of the k selected models. Let A be the minumum RMSEP-CV 
    observed between the k models, then:
    * discr(model i) = RMSEP-CV(model i) - A ; i = 1:k.
    Then, a weight is computed for each model i using a parameterized decreasing weighting function
    (function `winvs`). This returns a vector of model weights of length k, `w_emb`.   

4) The final prediction for an observation is the weighted (`w_emb`) average of the k predictions
    for this observation. 
    
5) The variable (`X` column) importance (`vi`) is the simple average of the model weights of all the models 
    (within the k models) that used this variable.  

# Examples
```julia
using Jchemo, JchemoData, DataFrames, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "tecator.jld2") 
@load db dat
@names dat
X = dat.X
Y = dat.Y 
wlst = names(X)
wl = parse.(Float64, wlst) 
ntot, p = size(X)
typ = Y.typ
namy = names(Y)[1:3]
plotsp(X, wl; xlabel = "Wavelength (nm)", ylabel = "Absorbance").f

s = typ .== "train"
Xtrain = X[s, :]
Ytrain = Y[s, namy]
Xtest = rmrow(X, s)
Ytest = rmrow(Y[:, namy], s)
ntrain = nro(Xtrain)
ntest = nro(Xtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

# Work on the j-th y-variable 
j = 2
nam = namy[j]
ytrain = Ytrain[:, nam]
ytest = Ytest[:, nam]
###### End

h = .2
#h = 1.
model = mwplsr(; h)
fit!(model, Xtrain, ytrain) 
fitm = model.fitm ;
@names fitm 

@head fitm.window            # range of each window 
@head centrw = fitm.centrw   # central point of each window

@head fitm.nlv_emb           # nb. of LVs selected for each window
tab(fitm.nlv_emb) 

@head w_emb = fitm.w_emb   # model weight for each window
@head vi = fitm.vi         # variable importances

res = predict(fitm, Xtest) ;
@names res 
@head pred = res.pred    # final predictions (weighted averages over the windiows)
size(res.predw)          # predictions for each window

@show rmsep(pred, ytest) 
plotxy(pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",
    ylabel = "Observed").f

xticks = collect(wl[1]:15:(wl[end]))
f, ax = plotxy(wl[centrw], w_emb; size = (800, 300), xticks, xlabel = "Window center (nm)", ylabel = "Model weight")
lines!(wl[centrw], w_emb; linewidth = .5)
f 

xticks = collect(wl[1]:15:(wl[end]))
f, ax = plotxy(wl, vi; size = (800, 300), xticks, xlabel = "Wavelength (nm)", ylabel = "Variable importance")
lines!(wl, vi; linewidth = .5)
f 
```
"""
mwplsr(; kwargs...) = JchemoModel(mwplsr, nothing, kwargs)

function mwplsr(X, Y; kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = pweight(ones(eltype(X), nro(X)))
    mwplsr(X, Y, weights; kwargs...)
end

function mwplsr(X::AbstractMatrix{Q}, Y::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; 
    kwargs...) where Q <: Float

    n, p = size(X)
    q = nco(Y)                    
    par = recovkw(ParMwplsr, kwargs).par
    @assert isodd(par.npoint) && par.npoint >= 1 "Argument 'npoint' must be an odd integer >= 1."
    
    nhwindow = Int((par.npoint - 1) / 2)  # half window
    rangetot = (nhwindow + 1):(p - nhwindow)
    centrw = collect(rangetot)
    nmod = length(rangetot)
    
    window = list(UnitRange{Int}, nmod)
    fitm_emb = list(Plsr, nmod)
    nlv_emb = list(Int, nmod)
    scor_emb = list(Q, nmod)
        
    j = 1   
    @inbounds for i in rangetot  # define each window
    
        # Define the window
        #i = rangetot[1]
        window[j] = (i - nhwindow):(i + nhwindow)
        vX = vcol(X, window[j])
        
        # Cross-validate to select the model
        segm = segmkf(n, par.Kcv; rep = par.repcv)
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
        res[u, :]

        # Fit the selected model        
        nlv_emb[j] = res.nlv[u]
        fitm_emb[j] = plskern(vX, Y, weights; nlv = nlv_emb[j]) #, scal = par.scal)
        scor_emb[j] = vscor[u]
                
        j = j + 1
    
    end

    # Model weights
    scor_best = minimum(scor_emb)
    d_emb = scor_emb .- scor_best   # model performance discrepancy
    criw = 3. ; squared = false
    w_emb = winvs(d_emb; h = par.h, criw, squared)
    wtot = sum(w_emb)
    @. w_emb /= wtot                # sum(w_emb) = 1

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
    
    Mwplsr(fitm_emb, nlv_emb, scor_emb, d_emb, w_emb, vi, window, centrw) 

end

function predict(object::Mwplsr, X)
    
    X = ensure_mat(X)
    Q = eltype(X)
    m = nro(X)
    q = length(object.fitm_emb[1].ymeans) 
    nmod = length(object.fitm_emb)

    predw = similar(X, m, q, nmod)
    @inbounds for i in eachindex(object.fitm_emb)
        predw[:, :, i] = predict(object.fitm_emb[i], vcol(X, object.window[i])).pred
    end

    pred = zeros(Q, m, q)
    @inbounds for k in 1:nmod
        pred = pred + object.w_emb[k] * predw[:, :, k] 
    end

    (pred = pred, predw)

end


