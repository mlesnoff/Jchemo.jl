"""
    occmwpls(; kwargs...)
    occmwpls(X; kwargs...)
One-class classification (OCC) by moving window Pls (MWPLS).
* `X` : Training (reference class) X-data (n, p).
* `Y` : Training (reference class) Y-data (n, q).
Keyword arguments:
* `npoint` : Total number of points in the sliding window (must be odd).
* `nlv` : Maximum nb. of latent variables (LVs) to consider in each Pls model.
* `K` : Number of folds in the K-fold cross validation used to select the nb. LVs for each window. 
* `rep` : Number of repetitions of the K-fold cross validation used to select the nb. LVs for each window.
* `typcut` : Type of cutoffs. Possible values are: `:std`, `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:std` or `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.
* `gamma` : Proportion of scaled SD in the consensus (see function `outsdod`).
* `scal` : Symbol defining the column scaling of `X` and `Y`. Possible values are: `:none` (default), `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

The function implements an Occ by moving (i.e., sliding) window Pls (Mwpls). This is the same principle as function `occmwpca` except
that a Pls {X, Y} is run on each window, instead of a Pca. For each window, the nb. LVs is selected by a repeated K-fold 
cross validation, see function `mwplsr`. 

# Examples
```julia
using Jchemo, JchemoData, JLD2, GLMakie, CairoMakie
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
conctrain = Ytrain.conc
yclatrain = Ytrain.typ
Xtest = Xp[s, :]
Ytest = Y[s, :]
yclatest = Ytest.typ 

#### Build the data used in the present example
# "EHH" = Training reference class (= target = 'in')
s = yclatrain .== "EHH"
Xtrain_ref = Xtrain[s, :]    
conctrain_ref = conctrain[s]
ntrain_ref = nro(Xtrain_ref)
# New reference observations ("EHH") to be predicted ==> should be predicted 'in'
s = yclatest .== "EHH"
Xnew_ref = Xtest[s, :] 
nnew_ref = nro(Xnew_ref)
# New observations 'out' ("PEE") to be predicted ==> should be predicted 'out'
s = yclatest .== "PEE"
Xnew_out = Xtest[s, :] 
nnew_out = nro(Xnew_out)
# Required to compute classification error rates (define the final class variable)
ntot = ntrain_ref + nnew_ref + nnew_out
(ntot = ntot, ntrain_ref, nnew_ref, nnew_out)
yref = fill("in", ntrain_ref)
ynew_ref = fill("in", nnew_ref)
ynew_out = fill("out", nnew_out)

#### Preliminary data description
# Fit a preliminary Pca model on the training reference data
nlv = 15
model = pcasvd(; nlv) 
#model = pcaout(; nlv) 
fit!(model, Xtrain_ref) 
fitm = model.fitm ;
res = summary(model, Xtrain_ref).explvarx 
plotgrid(res.nlv, res.pvar; step = 2, xlabel = "Nb. LVs", ylabel = "% Variance explained").f
Ttrain_ref = fitm.T
# Project the test observations in the fitted score space
Tnew_ref = transf(model, Xnew_ref)
Tnew_out = transf(model, Xnew_out)
#GLMakie.activate!()   # requires GLMakie
T = vcat(Ttrain_ref, Tnew_ref, Tnew_out)
group = vcat(fill("1-Train_ref", ntrain_ref), fill("2-New_ref", nnew_ref), fill("3-New_out", nnew_out))
lev = mlev(group)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
i = 1
plotxyz(T[:, i], T[:, i + 1], T[:, i + 2], group; color, leg_title = "Type of obs.", 
    xlabel = string("PC", i), ylabel = string("PC", i + 1), zlabel = string("PC", i + 2)).f
#### End

#### Fit the Occ model
npoint = 21
nlv = 15
pctvar = .98
typcut = :q ; alpha = .10
gamma = .5
#gamma = 0. # i.e., only OD is computed
model = occmwpls(; npoint, nlv, pctvar, typcut, alpha, gamma) 
fit!(model, Xtrain_ref, conctrain_ref)
fitm = model.fitm ;
@names fitm 

@head fitm.window            # range of each window
@head centrw = fitm.centrw   # central point of each window

@head fitm.nlv_emb           # nb. of LVs selected for each window
tab(fitm.nlv_emb)

@head fitm.d                 # outlierness of the training observations for each window 
                             # (1 row = 1 observation, 1 column = 1 window)

@head cutoff_d = fitm.cutoff_d    # cutoff (computed from d) of each window
@head pwout = fitm.pwout          # proportion of windows with outlierness d > cutoff_d, for each training observations 
cutoff_pwout = fitm.cutoff_pwout  # final cutoff computed from pwout 

d = fitm.d
tsp = .2 ; color = (:orange, tsp)
f, ax = plotsp(d, centrw; color, title = "Train", 
    xlabel = "Window center", ylabel = "Outlierness", label = "Train_ref")
lines!(ax, centrw, cutoff_d; color = :grey, linewidth = 2, label = "Cutoff")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f

tsp = .4 ; color = (:orange, tsp)
f = Figure(size = (450, 300)) 
ax = Axis(f[1, 1]; xticks = ([0], [""]), title = "Train", xlabel = "", ylabel = "pwout") 
rainclouds!(ax, fill(1, ntrain_ref), pwout; clouds = hist, jitter_width = .1, color, markersize = 10)
hlines!(ax, cutoff_pwout; color = :grey, linestyle = :dash, label = "Cutoff")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f

#### Predict the new observations 'ref'
res = predict(model, Xnew_ref) ;
@names res
@head pred = res.pred       # final predictions 'in/out'

@head dnew_ref = res.d      # predicted outlierness of the new observations for each window 
                            # (1 row = 1 observation, 1 column = 1 window)

@head pwoutnew_ref = res.pwout  # predicted proportion of windows with outlierness d > cutoff_d

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
dnew = copy(dnew_ref) ; pwoutnew = copy(pwoutnew_ref) ; nam = "ref"
#dnew = copy(dnew_out) ; pwoutnew = copy(pwoutnew_out) ; nam = "out"
m = nro(dnew)
tsp = .2 ; color = (:orange, tsp)
i = 1  # new observation to plot
f, ax = plotsp(d, centrw; color, xlabel = "Wavelength index",  ylabel = "Outlierness (SD-OD)", 
    label = "Train_ref")
lines!(ax, centrw, cutoff_d; color = :grey, linewidth = 2, label = "Cutoff")
lines!(ax, centrw, vrow(dnew, i); color = :blue, linewidth = .5, 
    label = string("An obs. in New_", nam))
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f

v = vcat(pwout, pwoutnew_ref, pwoutnew_out)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
groupnum = vcat(fill(1, ntrain_ref), fill(2, nnew_ref), fill(3, nnew_out))
cols = vcat(fill(color[1], ntrain_ref), fill(color[2], nnew_ref), fill(color[3], nnew_out))
f = Figure(size = (600, 300))
ax = Axis(f[1, 1]; xticks = (1:3, lev), xlabel = "", ylabel = "pwout") 
rainclouds!(ax, groupnum, v; clouds = hist, jitter_width = .1, color = cols, markersize = 10)
hlines!(ax, cutoff_pwout; color = :grey, linestyle = :dash, linewidth = 1, label = "Cutoff")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f
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
    cutoff_pwout = quantv(pwout, 1 - par.alpha)
    Occmwpls(d, cutoff_d, pwout, cutoff_pwout, fitm_emb, nlv_emb, fitm_occ, window, centrw)
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
    pred = [if pwout[i] <= object.cutoff_pwout "in" else "out" end for i in eachindex(pwout)]
    pred = reshape(pred, m, 1)
    (pred = pred, d, pwout)
end


