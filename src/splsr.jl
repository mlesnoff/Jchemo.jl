"""
    splsr(; kwargs...)
    splsr(X, Y; kwargs...)
    splsr(X, Y, weights::Weight; kwargs...)
    splsr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
Sparse partial least squares regression (Lê Cao et al. 2008)
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g., function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `meth` : Method used for the sparse thresholding. Possible values are: `:soft`, `:hard`. See thereafter.
* `nvar` : Nb. variables (`X`-columns) selected for each LV. Can be a single integer (i.e. same nb. 
    of variables for each LV), or a vector of length `nlv`.   
* `tol` : Only when q > 1; tolerance used in function `snipals_shen`. 
* `maxit` : Only when q > 1; maximum nb. of iterations used in function `snipals_shen`.    
* `scal` : Boolean. If `true`, each column of `X` and `Y` is scaled by its uncorrected standard deviation.    

Sparse partial least squares regression algorihm of Lê Cao et al. 2008, but with the fast 
"improved kernel algorithm #1" of Dayal & McGregor (1997) used instead Nipals (results are the same). 

In the present version of `splsr`, only the `X`-loading weights (not the `Y`-loading weights) are penalized. 
The function provides two thresholding methods: `:soft` and `:hard`, see function `spca` for description. 

In brief, to penalize the `X`-loading weights, the trick of Lê Cao et al. 2008 algorithm is to apply the 
sPCA-rSVD algorithm (Shen & Huang 2008) on matrix `Y'X` (instead of `X` in sparse PCA).

When `meth = :soft` the function returns the same results as function `spls` of the R package mixOmics 
(Lê Cao et al.) when regression mode and no sparseness on `Y` are specified.

The case `nvar = 1` corresponds to the Covsel regression method described in Roger et al 2011 (see also 
Höskuldsson 1992).

## References

Dayal, B.S., MacGregor, J.F., 1997. Improved PLS algorithms. Journal of Chemometrics 11, 73-85.

Höskuldsson, A., 1992. The H-principle in modelling with applications to chemometrics. Chemometrics 
and Intelligent Laboratory Systems, Proceedings of the 2nd Scandinavian Symposium on Chemometrics 14, 
139–153. https://doi.org/10.1016/0169-7439(92)80099-P

Lê Cao, K.-A., Rossouw, D., Robert-Granié, C., Besse, P., 2008. A Sparse PLS for Variable Selection 
when Integrating Omics Data. Statistical Applications in Genetics and Molecular Biology 7. 
https://doi.org/10.2202/1544-6115.1390

Kim-Anh Lê Cao, Florian Rohart, Ignacio Gonzalez, Sebastien Dejean with key contributors Benoit Gautier, Francois 
Bartolo, contributions from Pierre Monget, Jeff Coquery, FangZou Yao and Benoit Liquet. (2016). mixOmics: Omics Data 
Integration Project. R package version 6.1.1. 
https://www.bioconductor.org/packages/release/bioc/html/mixOmics.html

Package mixOmics on Bioconductor:
https://www.bioconductor.org/packages/release/bioc/html/mixOmics.html

Roger, J.M., Palagos, B., Bertrand, D., Fernandez-Ahumada, E., 2011. Covsel: Variable selection for highly 
multivariate and multi-response calibration: Application to IR spectroscopy. 
Chem. Lab. Int. Syst. 106, 216-223.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat
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
meth = :soft
#meth = :hard
nvar = 20
model = splsr(; nlv, meth, nvar) ;
fit!(model, Xtrain, ytrain)
@names model
fitm = model.fitm ;
@names fitm
@head fitm.T
@head fitm.W

fitm.niter

fitm.sellv
fitm.sel

coef(model)
coef(model; nlv = 3)

@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

res = predict(model, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", ylabel = "Observed").f    

res = summary(model, Xtrain) ;
@names res
z = res.explvarx
plotgrid(z.nlv, z.cumpvar; step = 2, xlabel = "Nb. LVs", ylabel = "Prop. Explained X-Variance").f
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
    @assert in([:soft; :hard])(par.meth) "Wrong value for argument 'meth'."
    Q = eltype(X)
    isa(Y, BitMatrix) ? Y = convert.(Q, Y) : nothing
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, par.nlv)
    nvar = par.nvar
    length(nvar) == 1 ? nvar = repeat([nvar], nlv) : nothing
    if par.meth == :soft 
        fthresh = thresh_soft
    elseif par.meth == :hard 
        fthresh = thresh_hard
    end
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
    ## XtY 
    rweight!(Y, weights.w)
    XtY = X' * Y
    YtX = XtY'
    ## Pre-allocation
    T = similar(X, n, nlv)
    W = similar(X, p, nlv)
    V = copy(W)
    R = copy(V)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    t   = similar(X, n)
    dt  = copy(t)   
    zp  = similar(X, p)
    w   = copy(zp)
    absw = copy(zp)
    r   = copy(zp)
    c   = similar(X, q)
    tmpXtY = similar(XtY)
    u = list(Int64, p)
    niter = ones(Int, nlv)
    sellv = list(Vector{Int}, nlv)
    @inbounds for a = 1:nlv
        if q == 1
            w .= vcol(XtY, 1)
            ## Sparsity
            nzeros = p - nvar[a]
            if nzeros > 0
                absw .= abs.(w)
                u .= sortperm(absw; rev = true)
                sel = u[1:nvar[a]]
                qt = minimum(absw[sel])
                lambda = maximum(absw[absw .< qt])
                w .= fthresh.(w, lambda)
            end
            ## End
            w ./= normv(w)
        else
            res = snipals_shen(YtX; meth = par.meth, nvar = nvar[a], tol = par.tol, maxit = par.maxit)
            w .= res.v
            niter[a] = res.niter
        end                                  
        r .= w
        if a > 1
            @inbounds for j = 1:(a - 1)
                r .-= dot(w, vcol(V, j)) .* vcol(R, j)    
            end
        end                   
        mul!(t, X, r)                 
        dt .= weights.w .* t          
        tt = dot(t, dt)               
        mul!(c, YtX, r)
        c ./= tt                      
        mul!(zp, X', dt)              
        XtY .-= mul!(tmpXtY, zp, c')
        YtX .= XtY'     
        V[:, a] .= zp ./ tt           
        T[:, a] .= t                  
        W[:, a] .= w
        R[:, a] .= r
        C[:, a] .= c
        TT[a] = tt
        sellv[a] = findall(abs.(w) .> 0)
     end
     sel = sort(unique(reduce(vcat, sellv)))
     Splsr(T, V, R, W, C, TT, xmeans, xscales, ymeans, yscales, weights, 
         niter,      # related to sparseness: q = 1 ==> 1 (no iteration), q > 1 ==> output of snipals_shen
         sellv, sel, # add compared to ::Plsr
         par)
end



