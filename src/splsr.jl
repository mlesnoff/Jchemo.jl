"""
    splsr(; kwargs...)
    splsr(X, Y; kwargs...)
    splsr(X, Y, weights::Weight; kwargs...)
    splsr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
Sparse partial least squares regression (Lê Cao et al. 2008)
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `meth` : Method used for the sparse thresholding. 
    Possible values are: `:soft`, `:softs`, 
    `:hard`. See thereafter.
* `nvar` : Only used if `meth = :soft` or `meth = :hard`.
    Nb. variables (`X`-columns) selected for each latent
    variable (LV). Can be a single integer (i.e. same nb. 
    of variables for each PC), or a vector of length `nlv`.   
* `delta` : Only used if `meth = :softs`. Constant used in function 
   `soft` for the thresholding on the loadings (after they are 
    standardized to their maximal absolute value). Must ∈ [0, 1].
    Higher is `delta`, stronger is the thresholding. 

Adaptation of the sparse partial least squares regression algorihm of 
Lê Cao et al. 2008. The fast "improved kernel algorithm #1" of 
Dayal & McGregor (1997) is used instead Nipals. 

In the present version of `splsr`, the sparse correction 
only concerns `X`. The function provides three methods of 
thresholding to compute the sparse `X`-loading weights w, 
see function `spca` for description (same principles). 
    
The case `meth = :soft` returns the same results as function `spls` of 
the R package mixOmics (Lê Cao et al.) with the regression mode (and without 
sparseness on `Y`).

The COVSEL regression method described in Roger et al 2011 (see also
Höskuldsson 1992) can be implemented by setting `meth = :hard` 
(or `meth = :soft`) and `nvar = 1`.

## References

Dayal, B.S., MacGregor, J.F., 1997. Improved PLS algorithms. 
Journal of Chemometrics 11, 73-85.

Höskuldsson, A., 1992. The H-principle in modelling with applications 
to chemometrics. Chemometrics and Intelligent Laboratory Systems, 
Proceedings of the 2nd Scandinavian Symposium on Chemometrics 14, 
139–153. https://doi.org/10.1016/0169-7439(92)80099-P

Lê Cao, K.-A., Rossouw, D., Robert-Granié, C., Besse, P., 2008. 
A Sparse PLS for Variable Selection when Integrating Omics Data. 
Statistical Applications in Genetics and Molecular Biology 7. 
https://doi.org/10.2202/1544-6115.1390

Kim-Anh Lê Cao, Florian Rohart, Ignacio Gonzalez, Sebastien Dejean 
with key contributors Benoit Gautier, Francois Bartolo, contributions 
from Pierre Monget, Jeff Coquery, FangZou Yao and Benoit Liquet. 
(2016). mixOmics: Omics Data Integration Project. R package 
version 6.1.1. https://CRAN.R-project.org/package=mixOmics

Package mixOmics on Bioconductor:
https://www.bioconductor.org/packages/release/bioc/html/mixOmics.html

Roger, J.M., Palagos, B., Bertrand, D., Fernandez-Ahumada, E., 2011. 
covsel: Variable selection for highly multivariate and multi-response 
calibration: Application to IR spectroscopy. 
Chem. Lab. Int. Syst. 106, 216-223.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = 15
meth = :soft ; nvar = 20
#meth = :hard ; nvar = 20
model = splsr(; nlv, meth, nvar) ;
fit!(model, Xtrain, ytrain)
pnames(model)
pnames(model.fitm)
@head model.fitm.T
@head model.fitm.W

coef(model)
coef(model; nlv = 3)

@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

res = predict(model, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

res = summary(model, Xtrain) ;
pnames(res)
z = res.explvarx
plotgrid(z.nlv, z.cumpvar; step = 2, xlabel = "Nb. LVs", 
    ylabel = "Prop. Explained X-Variance").f
```
""" 
splsr(; kwargs...) = JchemoModel(splsr, nothing, kwargs)

function splsr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    splsr(X, Y, weights; kwargs...)
end

function splsr(X, Y, weights::Weight; kwargs...)
    splsr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function splsr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
    par = recovkw(ParSplsr, kwargs).par
    @assert in([:hard ; :softs ; :soft])(par.meth) "Wrong value for argument 'meth'."
    @assert 0 <= par.delta <= 1 "Argument 'delta' must ∈ [0, 1]." 
    Q = eltype(X)
    isa(Y, BitMatrix) ? Y = convert.(Q, Y) : nothing
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, par.nlv)
    nvar = par.nvar
    length(nvar) == 1 ? nvar = repeat([nvar], nlv) : nothing
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)  
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        fcscale!(X, xmeans, xscales)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(X, xmeans)
        fcenter!(Y, ymeans)
    end
    D = Diagonal(weights.w)
    XtY = X' * (D * Y)                   # = Xd' * Y = X' * D * Y  (Xd = D * X   Very costly!!)
    #XtY = X' * (weights .* Y)           # Can create OutOfMemory errors for very large matrices
    ## Pre-allocation
    T = similar(X, n, nlv)
    W = similar(X, p, nlv)
    P = copy(W)
    R = copy(P)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    t   = similar(X, n)
    dt  = copy(t)   
    zp  = similar(X, p)
    w   = copy(zp)
    absw = copy(zp)
    absw_stand = copy(zp)
    theta = copy(zp)
    r   = copy(zp)
    c   = similar(X, q)
    tmp = similar(XtY) # = XtY_approx
    sellv = list(Vector{Int}, nlv)
    @inbounds for a = 1:nlv
        if q == 1
            w .= vcol(XtY, 1)
            absw .= abs.(w)
            if par.meth == :soft
                nrm = p - nvar[a]
                if nrm > 0
                    sel = sortperm(absw; rev = true)[1:nvar[a]]
                    wmax = w[sel]
                    w .= zeros(Q, p)
                    w[sel] .= wmax
                    zdelta = maximum(sort(absw)[1:nrm])
                    w .= soft.(w, zdelta)
                end
            elseif par.meth == :softs
                absw_max = maximum(absw)
                absw_stand .= absw / absw_max
                theta .= max.(0, absw_stand .- par.delta) 
                w .= sign.(w) .* theta * absw_max 
            elseif par.meth == :hard
                sel = sortperm(absw; rev = true)[1:nvar[a]]
                wmax = w[sel]
                w .= zeros(Q, p)
                w[sel] .= wmax
            end
            ## End
            w ./= normv(w)
        else
            if par.meth == :softs
                w .= snipals_softs(XtY'; kwargs...).v
            else
                par.nvar = nvar[a]
                if par.meth == :soft
                    w .= snipals_soft(XtY'; kwargs...).v
                else
                    w .= snipals_h(XtY'; kwargs...).v
                end
            end
        end                                  
        r .= w
        if a > 1
            @inbounds for j = 1:(a - 1)
                r .-= dot(w, vcol(P, j)) .* vcol(R, j)    
            end
        end                   
        mul!(t, X, r)                 
        dt .= weights.w .* t          
        tt = dot(t, dt)               
        mul!(c, XtY', r)
        c ./= tt                      
        mul!(zp, X', dt)              
        XtY .-= mul!(tmp, zp, c')     
        P[:, a] .= zp ./ tt           
        T[:, a] .= t                  
        W[:, a] .= w
        R[:, a] .= r
        C[:, a] .= c
        TT[a] = tt
        sellv[a] = findall(abs.(w) .> 0)
     end
     sel = unique(reduce(vcat, sellv))
     Splsr(T, P, R, W, C, TT, xmeans, xscales, ymeans, yscales, weights, nothing, 
         sellv, sel,  # add compared to ::Plsr
         par)
end



