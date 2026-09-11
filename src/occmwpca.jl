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

The function implements an Occ by moving window Pca (Mwpca; e.g., Lennox et al 2001, Jeng 2010), as follows.

1) The full range of the X-columns is divided into sliding windows of `npoint` columns. The center of each 
    window is offset of one column from the center of the previous window. 

2) On each window:
    * a Pca model with `nlv` LVs (principal components) is fitted, and the minimum nb. of LVs that explains 
        at least a proportion `pctvar` of the total variance (of the window) is retained.

    * Function `occsdod` (SD-OD outlierness consensus) is applied on the fitted Pca model (with the retained 
        nb. LVs) to compute the outlierness (d) of the training observations for the window, and a cutoff is 
        determined (see function `occsdod`).
    
    * New observations are predicted (for the considered window) from this fitted `occsdod` model and their 
        outlierness d are computed.
    
3) For each observation (training or new), the proportion of windows for which the observation is 
    predicted as an outlier for d (i.e., SD-OD `occsdod` consensus > cutoff) is computed. A second
    (and final) cutoff is then determined on the n proportions computed on the training. 
    
4) Observations that have a higher proportion of windows with outlier d than this second cutoff 
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

#### Build the data used in the example
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

# Only used to compute classification error rates
ntot = nref + nnew_ref + nnew_out
(ntot = ntot, nref, nnew_ref, nnew_out)
yref = fill("in", nref)
ynew_ref = fill("in", nnew_ref)
ynew_out = fill("in", nnew_out)

#### Fit a preliminary Pca model on the training reference data
nlv = 15
model0 = pcasvd(; nlv) 
#model0 = pcaout(; nlv) 
fit!(model0, Xref) 
fitm0 = model0.fitm ;
res = summary(model0, Xref).explvarx 
plotgrid(res.nlv, res.pvar; step = 2, xlabel = "Nb. LVs", ylabel = "% Variance explained").f
Tref = fitm0.T

#### To describe the data, 
#### project the test observations in the fitted score space
Tnew_ref = transf(model0, Xnew_ref)
Tnew_out = transf(model0, Xnew_out)
#GLMakie.activate!()   # requires GLMakie
T = vcat(Tref, Tnew_ref, Tnew_out)
group = vcat(fill("1-Train (ref)", nref), fill("2-New_ref", nnew_ref), fill("3-New_out", nnew_out))
lev = mlev(group)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
i = 1
plotxyz(T[:, i], T[:, i + 1], T[:, i + 2], group; color, leg_title = "Type of obs.", 
    xlabel = string("PC", i), ylabel = string("PC", i + 1), zlabel = string("PC", i + 2)).f

#### Fit the Occ model
nlv = 10
pctvar = .98
typcut = :q ; alpha = .10
gamma = .5
#gamma = 0. # only OD 
npoint = 11
model = occmwpca(; nlv, pctvar, typcut, alpha, gamma, npoint) 
fit!(model, Xref)
fitm = model.fitm ;
@names fitm 
@head fitm.rangemod          # range of each sliding windows
@head xsel = fitm.xsel       # central point of each sliding windows
@head fitm.nlv_emb           # nb. of LVs retained for each sliding window
tab(fitm.nlv_emb)
@head d = fitm.d             # outlierness of the training observations for each sliding window (1 column = 1 window)
@head cutoff = fitm.cutoff   # cutoff(computed from d) of each sliding window
@head pxout = fitm.pxout     # proportion of windows with outlierness d > cutoff, for the training observations 
cut_pxout = fitm.cut_pxout   # final cutoff computed from pxout 

tsp = .2 ; color = (:orange, tsp)
f, ax = plotsp(d, xsel; color, title = "Train", 
    xlabel = "Wavelength index", ylabel = "Outlierness (SD-OD)", label = "Train")
lines!(ax, xsel, cutoff; color = :grey, linewidth = 2, label = "Cutoff")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f

tsp = .4 ; color = (:orange, tsp)
f = Figure(size = (450, 300)) 
ax = Axis(f[1, 1]; xticks = ([1], ["Train"]), xlabel = "", ylabel = "pxout") 
rainclouds!(ax, fill(1, nref), pxout; clouds = hist, jitter_width = .1, color, markersize = 10)
hlines!(ax, cut_pxout; color = :grey, linestyle = :dash, label = "Cutoff")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f

#### Predict the new reference observations
res = predict(model, Xnew_ref) ;
@names res
@head pred = res.pred           # final predictions in/out
@head dnew_ref = res.d          # predicted outlierness d for each sliding window (1 column = 1 window)
@head pxoutnew_ref = res.pxout  # predicted proportion of windows with outlierness d > cutoff
tab(pred)
errp(pred, ynew_ref)
conf(pred, ynew_ref).cnt

#### Predict the new observations 'out'
res = predict(model, Xnew_out) ;
@names res
@head pred = res.pred 
@head dnew_out = res.d
@head pxoutnew_out = res.pxout 
tab(pred)
errp(pred, ynew_out)
conf(pred, ynew_out).cnt

dnew = copy(dnew_ref) ; pxoutnew = copy(pxoutnew_ref) ; title = "New_ref"
#dnew = copy(dnew_out) ; pxoutnew = copy(pxoutnew_out) ; title = "New_out"
m = nro(dnew)
tsp = .1 ; color = (:orange, tsp)
i = 1  # new observation to plot
f, ax = plotsp(d, xsel; color, title, 
    xlabel = "Wavelength index", ylabel = "Outlierness (SD-OD)", label = "Train")
lines!(ax, xsel, cutoff; color = :grey, linewidth = 2, label = "Cutoff")
lines!(ax, xsel, vrow(dnew, i); color = :blue, linewidth = .5, label = "New obs.")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f

d = vcat(pxout, pxoutnew_ref, pxoutnew_out)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
groupnum = vcat(fill(1, nref), fill(2, nnew_ref), fill(3, nnew_out))
cols = vcat(fill(color[1], nref), fill(color[2], nnew_ref), fill(color[3], nnew_out))
CairoMakie.activate!()
f = Figure(size = (600, 300))
ax = Axis(f[1, 1]; xticks = (1:3, lev), xlabel = "", ylabel = "pxout") 
rainclouds!(ax, groupnum, d; clouds = hist, jitter_width = .1, color = cols, markersize = 10)
hlines!(ax, cut_pxout; color = :grey, linestyle = :dash, linewidth = 1, label = "cutoff")
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
    @assert isodd(par.npoint) && par.npoint >= 1 "Argument 'npoint' must an odd integer >= 1."
    nhwindow = Int((par.npoint - 1) / 2)  # half window
    range_sel = (nhwindow + 1):(p - nhwindow)
    xsel = collect(range_sel)
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
        fitm_emb[j] = par.fun(vX; par.nlv)
        vres = summary(fitm_emb[j], vX).explvarx
        nlv_emb[j] = (1:par.nlv)[vres.cumpvar .> par.pctvar][1]
        fitm_occ[j] = occsdod(fitm_emb[j], vX; nlv = nlv_emb[j], typcut = par.typcut, 
            cri = par.cri, alpha = par.alpha, gamma = par.gamma)
        d[:, j] = fitm_occ[j].d.d 
        j = j + 1
    end
    #j = 1 ; summary(fitm_emb[j], vcol(X, rangemod[j])).explvarx
    cutoff = colquant(d, 1 - par.alpha)
    pxout = rowsum(Q.(d .> cutoff')) / nmodel
    cut_pxout = quantv(pxout, 1 - par.alpha)
    Occmwpca(fitm_emb, nlv_emb, fitm_occ, d, cutoff, pxout, cut_pxout, rangemod, xsel)
end

function predict(object::Occmwpca, X)
    X = ensure_mat(X)
    Q = eltype(X)
    m = nro(X)
    nmodel = length(object.rangemod)
    d = similar(X, m, nmodel)
    @inbounds for j in eachindex(object.rangemod)
        d[:, j] = predict(object.fitm_occ[j], vcol(X, object.rangemod[j])).d.d
    end
    pxout = rowsum(Q.(d .> object.cutoff')) / nmodel
    pred = [if pxout[i] <= object.cut_pxout "in" else "out" end for i in eachindex(pxout)]
    pred = reshape(pred, m, 1)
    (pred = pred, d, pxout)
end


