"""
    occmwpca(; kwargs...)
    occmwpca(X; kwargs...)
One-class classification (OCC) by moving window Pca (MWPCA).
* `X` : Training (reference class) X-data (n, p).
Keyword arguments:
* `fun` : Function used to fit the Pca models (by default: `pcasvd`).
* `nlv` : Maximum nb. of latent variables (LVs) to consider in each Pca model.
* `pctvar` : Minimum proportion (within ]0, 1]) of explained variance to consider in each Pca model.
* `typcut` : Type of cutoffs. Possible values are: `:std`, `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:std` or `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.
* `gamma` : Proportion of scaled SD in the consensus (see function `outsdod`).
* `npoint` : Total number of points in the sliding window (must be odd).

The function implements an Occ by moving window Pca (Mwpca) (e.g., Lennox et al 2001, Jeng 2010), as follows.

1) The full range of the X-columns is divided into sliding windows of `npoint` columns. The center of each 
    window is moved of one column from the previous window. 

2) On each window:
    * a Pca model with `nlv` LVs (principal components) is fitted, and the minimum nb. of LVs that explains 
        at least a proportion `pctvar` of the total variance (of the window) is retained.

    * Function `occsdod` (SD-OD outlierness consensus) is applied on the fitted Pca model (with the retained 
        nb. LVs) to compute the outlierness (d) of the training observations for the window, and a cutoff is 
        determined (see function `occsdod`).
    
    * New observations are predicted (for the considered window) from this fitted `occsdod` model and their 
        outlierness d are computed.
    
3) For each observation (training or new), the proportion of windows for which the observation is 
    predicted as an outlier for d (i.e., the SD-OD `occsdod` consensus > cutoff) is computed. A second
    (and final) cutoff is then determined on the n proportions computed on the training. 
    
4) Observations that have a higher proportion of windows with outliernes d higher than this second cutoff 
    are classified as 'out'. Others are classified as 'in'.

This version of the function is different from the approach proposed by Fernández Pierna et al (2016) 
in two ways. First, it is not a local (KNN) approach. Second, for each window, the SD-OD outlierness is 
computed from the full window (instead of only on the central point of the window).

See function `outsdod` for other details, and the examples below for the outputs.

# References
Fernández Pierna, J.A., Vincke, D., Baeten, V., Grelet, C., Dehareng, F., Dardenne, P., 2016. 
Use of a multivariate moving window PCA for the untargeted detection of contaminants in agro-food products, 
as exemplified by the detection of melamine levels in milk using vibrational spectroscopy. 
Chemometrics and Intelligent Laboratory Systems 152, 157–162. 
https://doi.org/10.1016/j.chemolab.2015.10.016

Jeng, J.-C., 2010. Adaptive process monitoring using efficient recursive PCA and moving window
PCA algorithms. Journal of the Taiwan Institute of Chemical Engineers, Festschrift Issue 41, 475–481.
 https://doi.org/10.1016/j.jtice.2010.03.015

Lennox, B., Montague, G. a., Hiden, H. g., Kornfeld, G., Goulding, P. r., 2001. Process monitoring 
of an industrial fed-batch fermentation. Biotechnology and Bioengineering 74, 125–135. 
https://doi.org/10.1002/bit.1102

# Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/challenge2018.jld2") 
@load db dat
@names dat
X = dat.X    
Y = dat.Y
model = savgol(npoint = 21, deriv = 2, degree = 3)
fit!(model, X) 
Xp = transf(model, X) 
s = Bool.(Y.test)
Xtrain = rmrow(Xp, s)
Ytrain = rmrow(Y, s)
yclatrain = Ytrain.typ
Xtest = Xp[s, :]
Ytest = Y[s, :]
yclatest = Ytest.typ 

#### Build the data used in the present example
# "EHH" = Training reference class (= target = 'in')
s = yclatrain .== "EHH"
Xref = Xtrain[s, :]    
nref = nro(Xref)
# New reference observations ("EHH") to be predicted ==> should be predicted 'in'
s = yclatest .== "EHH"
Xnew_ref = Xtest[s, :] 
nnew_ref = nro(Xnew_ref)
# New observations 'out' ("PEE") to be predicted ==> should be predicted 'out'
s = yclatest .== "PEE"
Xnew_out = Xtest[s, :] 
nnew_out = nro(Xnew_out)
# Required to compute classification error rates
ntot = nref + nnew_ref + nnew_out
(ntot = ntot, nref, nnew_ref, nnew_out)
yref = fill("in", nref)
ynew_ref = fill("in", nnew_ref)
ynew_out = fill("in", nnew_out)

#### Preliminary data description
# Fit a preliminary Pca model on the training reference data
nlv = 15
model = pcasvd(; nlv) 
#model = pcaout(; nlv) 
fit!(model, Xref) 
fitm = model.fitm ;
res = summary(model, Xref).explvarx 
plotgrid(res.nlv, res.pvar; step = 2, xlabel = "Nb. LVs", ylabel = "% Variance explained").f
Tref = fitm.T
# Project the test observations in the fitted score space)
Tnew_ref = transf(model, Xnew_ref)
Tnew_out = transf(model, Xnew_out)
#GLMakie.activate!()   # requires GLMakie
T = vcat(Tref, Tnew_ref, Tnew_out)
group = vcat(fill("1-Train_ref", nref), fill("2-New_ref", nnew_ref), fill("3-New_out", nnew_out))
lev = mlev(group)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
i = 1
plotxyz(T[:, i], T[:, i + 1], T[:, i + 2], group; color, leg_title = "Type of obs.", 
    xlabel = string("PC", i), ylabel = string("PC", i + 1), zlabel = string("PC", i + 2)).f
#### End

#### Fit the Occ model
npoint = 11
nlv = 10
pctvar = .98
typcut = :q ; alpha = .10
gamma = .5
#gamma = 0. # i.e., only OD is computed
model = occmwpca(; npoint, nlv, pctvar, typcut, alpha, gamma) 
fit!(model, Xref)
fitm = model.fitm ;
@names fitm 

@head fitm.window            # range of each sliding windows
@head centrw = fitm.centrw   # central point of each sliding windows

@head fitm.nlv_emb           # nb. of Pca LVs selected for each sliding window
tab(fitm.nlv_emb)

@head fitm.d                 # outlierness of the training observations for each sliding window 
                             # (1 row = 1 observation, 1 column = 1 window)

@head cutoff = fitm.cutoff   # cutoff (computed from d) of each sliding window
@head pwout = fitm.pwout     # proportion of windows with outlierness d > cutoff, for each training observations 
cut_pwout = fitm.cut_pwout   # final cutoff computed from pwout 

d = fitm.d
tsp = .2 ; color = (:orange, tsp)
f, ax = plotsp(d, centrw; color, title = "Train", 
    xlabel = "Window center", ylabel = "Outlierness (SD-OD)", label = "Train_ref")
lines!(ax, centrw, cutoff; color = :grey, linewidth = 2, label = "Cutoff")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f

tsp = .4 ; color = (:orange, tsp)
f = Figure(size = (450, 300)) 
ax = Axis(f[1, 1]; xticks = ([1], ["Train_ref"]), xlabel = "", ylabel = "pwout") 
rainclouds!(ax, fill(1, nref), pwout; clouds = hist, jitter_width = .1, color, markersize = 10)
hlines!(ax, cut_pwout; color = :grey, linestyle = :dash, label = "Cutoff")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f

#### Predict the new observations 'ref'
res = predict(model, Xnew_ref) ;
@names res
@head pred = res.pred           # final predictions 'in/out'
@head dnew_ref = res.d          # predicted outlierness d for each sliding window (1 column = 1 window)
@head pwoutnew_ref = res.pwout  # predicted proportion of windows with outlierness d > cutoff
tab(pred)
errp(pred, ynew_ref)
conf(pred, ynew_ref).cnt

#### Predict the new observations 'out'
res = predict(model, Xnew_out) ;
@names res
@head pred = res.pred 
@head dnew_out = res.d
@head pwoutnew_out = res.pwout 
tab(pred)
errp(pred, ynew_out)
conf(pred, ynew_out).cnt

d = fitm.d
dnew = copy(dnew_ref) ; pwoutnew = copy(pwoutnew_ref) ; nam = "(ref)"
#dnew = copy(dnew_out) ; pwoutnew = copy(pwoutnew_out) ; nam = "(out)"
m = nro(dnew)
tsp = .2 ; color = (:orange, tsp)
i = 1  # new observation to plot
f, ax = plotsp(d, centrw; color, 
    xlabel = "Wavelength index", ylabel = "Outlierness (SD-OD)", label = "Train_ref")
lines!(ax, centrw, cutoff; color = :grey, linewidth = 2, label = "Cutoff")
lines!(ax, centrw, vrow(dnew, i); color = :blue, linewidth = .5, label = "A new obs. $nam")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f

v = vcat(pwout, pwoutnew_ref, pwoutnew_out)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
groupnum = vcat(fill(1, nref), fill(2, nnew_ref), fill(3, nnew_out))
cols = vcat(fill(color[1], nref), fill(color[2], nnew_ref), fill(color[3], nnew_out))
CairoMakie.activate!()
f = Figure(size = (600, 300))
ax = Axis(f[1, 1]; xticks = (1:3, lev), xlabel = "", ylabel = "pwout") 
rainclouds!(ax, groupnum, v; clouds = hist, jitter_width = .1, color = cols, markersize = 10)
hlines!(ax, cut_pwout; color = :grey, linestyle = :dash, linewidth = 1, label = "cutoff")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f
```
"""
occmwpca(; kwargs...) = JchemoModel(occmwpca, nothing, kwargs)

function occmwpca(X; kwargs...)
    X = ensure_mat(X)
    n, p = size(X)                    
    Q = eltype(X)    
    par = recovkw(ParOccmwpca{Q}, kwargs).par
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
        vX = vcol(X,window[j])
        fitm_emb[j] = par.fun(vX; par.nlv)
        vres = summary(fitm_emb[j], vX).explvarx
        nlv_emb[j] = (1:par.nlv)[vres.cumpvar .> par.pctvar][1]
        fitm_occ[j] = occsdod(fitm_emb[j], vX; nlv = nlv_emb[j], typcut = par.typcut, 
            cri = par.cri, alpha = par.alpha, gamma = par.gamma)
        d[:, j] = fitm_occ[j].d.d 
        j = j + 1
    end
    cutoff = colquant(d, 1 - par.alpha)
    pwout = rowsum(Q.(d .> cutoff')) / nmod
    cut_pwout = quantv(pwout, 1 - par.alpha)
    Occmwpca(fitm_emb, nlv_emb, fitm_occ, d, cutoff, pwout, cut_pwout,window, centrw)
end

function predict(object::Occmwpca, X)
    X = ensure_mat(X)
    Q = eltype(X)
    m = nro(X)
    nmod = length(object.window)
    d = similar(X, m, nmod)
    @inbounds for j in eachindex(object.window)
        d[:, j] = predict(object.fitm_occ[j], vcol(X, object.window[j])).d.d
    end
    pwout = rowsum(Q.(d .> object.cutoff')) / nmod
    pred = [if pwout[i] <= object.cut_pwout "in" else "out" end for i in eachindex(pwout)]
    pred = reshape(pred, m, 1)
    (pred = pred, d, pwout)
end


