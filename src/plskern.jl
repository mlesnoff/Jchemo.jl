"""
    plskern(; kwargs...)
    plskern(X, Y; kwargs...)
    plskern(X, Y, weights::Weight; kwargs...)
    plskern!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
Partial least squares regression (PLSR) with the "improved kernel algorithm #1" (Dayal & McGegor, 1997).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` and `Y` is scaled by its uncorrected 
    standard deviation.
    
About the row-weighting in PLS algorithms (`weights`): see in particular Schaal et al. 2002, 
Siccard & Sabatier 2006, Kim et al. 2011, and Lesnoff et al. 2020. 

## References
Dayal, B.S., MacGregor, J.F., 1997. Improved PLS algorithms. Journal of Chemometrics 11, 73-85.

Kim, S., Kano, M., Nakagawa, H., Hasebe, S., 2011. Estimation of active pharmaceutical ingredients 
content using locally weighted partial least squares and statistical wavelength selection. Int. 
J. Pharm., 421, 269-274.

Lesnoff, M., Metz, M., Roger, J.M., 2020. Comparison of locally weighted PLS strategies for regression
and discrimination on agronomic NIR Data. Journal of Chemometrics. e3209. 
https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.3209

Schaal, S., Atkeson, C., Vijayamakumar, S. 2002. Scalable techniques from nonparametric statistics 
for the real time robot learning. Applied Intell., 17, 49-60.

Sicard, E. Sabatier, R., 2006. Theoretical framework for local PLS1 regression and application to a 
rainfall dataset. Comput. Stat. Data Anal., 51, 1393-1410.

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
model = plskern(; nlv) ;
#model = plsnipals(; nlv) ;
#model = plswold(; nlv) ;
#model = plsrosa(; nlv) ;
#model = plssimp(; nlv) ;
fit!(model, Xtrain, ytrain)
@names model
@names model.fitm
@head model.fitm.T

coef(model)
coef(model; nlv = 3)

@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

res = predict(model, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",  
    ylabel = "Observed").f    

res = predict(model, Xtest; nlv = 1:2)
@head res.pred[1]
@head res.pred[2]

res = summary(model, Xtrain) ;
@names res
z = res.explvarx
plotgrid(z.nlv, z.cumpvar; step = 2, xlabel = "Nb. LVs", ylabel = "Prop. Explained X-Variance").f
```
""" 
plskern(; kwargs...) = JchemoModel(plskern, nothing, kwargs)

function plskern(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plskern(X, Y, weights; kwargs...)
end

function plskern(X, Y, weights::Weight; kwargs...)
    plskern!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function plskern!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
    par = recovkw(ParPlsr, kwargs).par
    ## Specific for Plsda functions
    Q = eltype(X)
    isa(Y, BitMatrix) ? Y = convert.(Q, Y) : nothing
    ## End
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, maximum(par.nlv)) # the use of 'maximum' is required for plsravg 
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
    fweight!(Y, weights.w)
    XtY = X' * Y
    ## Old
    ## D = Diagonal(weights.w)
    ## XtY = X' * (D * Y)    # Xd = D * X   Very costly!!
    ## Pre-allocation
    T = similar(X, n, nlv)
    W = similar(X, p, nlv)
    V = copy(W)
    R = copy(V)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    t   = similar(X, n)
    dt  = similar(X, n)   
    v  = similar(X, p)
    w   = similar(X, p)
    r   = similar(X, p)
    c   = similar(X, q)
    tmpXtY = similar(XtY) # = XtY_approx
    # End
    @inbounds for a = 1:nlv
        if q == 1
            w .= vcol(XtY, 1)
            w ./= normv(w)
        else
            w .= svd(XtY).U[:, 1]
        end                                  
        r .= w
        if a > 1
            @inbounds for j = 1:(a - 1)
                r .-= dot(w, vcol(V, j)) .* vcol(R, j)    
            end
        end                   
        mul!(t, X, r)                 # t = X * r
        dt .= weights.w .* t            # dt = D * t
        tt = dot(t, dt)               # tt = t' * dt = t' * D * t 
        mul!(c, XtY', r)
        c ./= tt                      # c = XtY' * r / tt
        mul!(v, X', dt)               # v = (D * X)' * t = X' * (D * t)
        XtY .-= mul!(tmpXtY, v, c')   # XtY = XtY - v * c' ; deflation of the kernel matrix 
        V[:, a] .= v ./ tt            # ==> the metric applied to covariance is applied outside the loop,
        T[:, a] .= t                  # conversely to other algorithms such as nipals
        W[:, a] .= w
        R[:, a] .= r
        C[:, a] .= c
        TT[a] = tt
    end
    Plsr(T, V, R, W, C, TT, xmeans, xscales, ymeans, yscales, weights, nothing, par)
end

""" 
    transf(object::Union{Plsr, Splsr}, X; nlv = nothing)
Compute latent variables (LVs = scores) from a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Union{Plsr, Splsr}, X; nlv = nothing)
    X = ensure_mat(X)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    ## Could be fcscale! but changes X. If too heavy ==> Makes summary!
    fcscale(X, object.xmeans, object.xscales) * vcol(object.R, 1:nlv)
end

"""
    coef(object::Union{Plsr, Pcr, Splsr}; nlv = nothing)
Compute the b-coefficients of a LV model.
* `object` : The fitted model.
* `nlv` : Nb. LVs to consider.

For a model fitted from X(n, p) and Y(n, q), the returned 
object `B` is a matrix (p, q). If `nlv` = 0, `B` is a matrix 
of zeros. The returned object `int` is the intercept.
""" 
function coef(object::Union{Plsr, Splsr}; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    theta = vcol(object.C, 1:nlv)'  # coeffs regression of Y on T
    Dy = Diagonal(object.yscales)
    ## To not use for Spcr (R not computed; while for Pcr, R = V)
    B =  fweight(vcol(object.R, 1:nlv), 1 ./ object.xscales) * theta * Dy
    ## In 'int': No correction is needed, since ymeans, xmeans and B are in the original scale 
    int = object.ymeans' .- object.xmeans' * B
    ## End
    (B = B, int = int)
end

"""
    predict(object::Union{Plsr, Pcr, Splsr}, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Union{Plsr, Splsr}, X; nlv = nothing)
    X = ensure_mat(X)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(a, minimum(nlv)):min(a, maximum(nlv))
    le_nlv = length(nlv)
    pred = list(Matrix{eltype(X)}, le_nlv)
    @inbounds for i in eachindex(nlv)
        coefs = coef(object; nlv = nlv[i])
        pred[i] = coefs.int .+ X * coefs.B  # try muladd(X, coefs.B, coefs.int)
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end

"""
    summary(object::Union{Plsr, Splsr}, X)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to 
    fit the model.
""" 
function Base.summary(object::Union{Plsr, Splsr}, X)
    X = ensure_mat(X)
    n, nlv = size(object.T)
    X = fcscale(X, object.xmeans, object.xscales)
    ## Could be fcscale! but changes X. If too heavy ==> Makes summary!
    sstot = frob2(X, object.weights)       # = sum(object.weights.w' * X.^2)
    tt = object.TT
    tt_adj = (colnorm(object.V).^2) .* tt  # tt_adj[a] = p[a]'p[a] * tt[a]
    xvar = tt_adj / n    
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)     
    (explvarx = explvarx,)
end
