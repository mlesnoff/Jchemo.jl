"""
    plskern(; kwargs...)
    plskern(X, Y; kwargs...)
    plskern(X, Y, weights::ProbabilityWeights; kwargs...)
    plskern!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::ProbabilityWeights; kwargs...)
Partial least squares regression (PLSR) with the "improved kernel algorithm #1" (Dayal & McGegor, 1997).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` and `Y` is scaled by its uncorrected standard deviation.
    
About the row-weighting in PLS algorithms (`weights`): see in particular Schaal et al. 2002, Siccard & Sabatier 
2006, Kim et al. 2011, and Lesnoff et al. 2020. 

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
fitm = model.fitm ;
@names fitm

@head transf(model, Xtrain)
@head fitm.T

@head transf(model, Xtest)
@head transf(model, Xtest, 3)  # first 3 LVs

coef(model)
coef(model, 3)  # b-coefs for the 3-LV model

res = predict(model, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

res = predict(model, Xtest, 1:2) # predictions with 1 and 2 LVs
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
    weights = pweight(ones(Q, nro(X)))
    plskern(X, Y, weights; kwargs...)
end

function plskern(X, Y, weights::ProbabilityWeights; kwargs...)
    plskern!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function plskern!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::ProbabilityWeights; kwargs...)
    par = recovkw(ParPlsr, kwargs).par
    Q = eltype(X)
    Y = handle_bitmatrix(Q, Y)  # for DA functions
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, par.nlv)
    par.nlv = nlv
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
    fweightr!(Y, weights.values)
    XtY = X' * Y
    ## Old
    ## D = Diagonal(weights.values)
    ## XtY = X' * (D * Y)    # Xd = D * X   Very costly!!
    ## Pre-allocation
    T  = similar(X, n, nlv)
    W  = similar(X, p, nlv)
    V  = similar(W)
    R  = similar(W)
    C  = similar(X, q, nlv)
    TT = similar(X, nlv)
    t  = similar(X, n)
    dt = similar(t)   
    v  = similar(X, p)
    w  = similar(v)
    r  = similar(v)
    c  = similar(X, q)
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
        @. dt = weights.values * t    # dt = D * t
        tt = dot(t, dt)               # tt = t' * dt = t' * D * t 
        mul!(c, XtY', r)
        c ./= tt                      # c = XtY' * r / tt
        mul!(v, X', dt)               # v = (D * X)' * t = X' * (D * t)
        XtY .-= mul!(tmpXtY, v, c')   # XtY = XtY - v * c' ; deflation of the kernel matrix 
        @. V[:, a] = v / tt           # ==> the metric applied to covariance is applied outside the loop,
        T[:, a] .= t                  # conversely to other algorithms such as nipals
        W[:, a] .= w
        R[:, a] .= r
        C[:, a] .= c
        TT[a] = tt
    end
    Plsr(T, V, R, W, C, TT, xmeans, xscales, ymeans, yscales, weights, nothing, par)
end

""" 
    transf(object::Union{Plsr, Splsr}, X)
    transf(object::Union{Plsr, Splsr}, X, nlv::Int)
Compute latent variables (LVs; = scores) from a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Union{Plsr, Splsr}, X)
    X = ensure_mat(X)
    ## Could be fcscale! but would change X. If too heavy ==> Makes summary!
    fcscale(X, object.xmeans, object.xscales) * object.R
end

function transf(object::Union{Plsr, Splsr}, X, nlv::Int)
    X = ensure_mat(X)
    nlv = min(nlv, object.par.nlv)
    ## Could be fcscale! but would change X. If too heavy ==> Makes summary!
    fcscale(X, object.xmeans, object.xscales) * vcol(object.R, 1:nlv)
end

"""
    coef(object::Union{Plsr, Splsr})
    coef(object::Union{Plsr, Splsr}, nlv::Int)
Compute the b-coefficients of a LV model.
* `object` : The fitted model.
* `nlv` : Nb. LVs to consider.

For a model fitted from X (n, p) and Y (n, q), the returned object `B` is a matrix (p, q). 
If `nlv` = 0, `B` is a matrix of zeros. The returned object `int` is the intercept.
""" 
function coef(object::Union{Plsr, Splsr})
    theta = object.C'  # regression coefs of Y on T
    Dy = Diagonal(object.yscales)
    ## To not use for Spcr (R not computed; while for Pcr, R = V)
    B = fweightr(object.R, 1 ./ object.xscales) * theta * Dy
    ## In 'int': No correction is needed, since ymeans, xmeans and B are in the original scale 
    int = object.ymeans' .- object.xmeans' * B
    (B = B, int)
end

function coef(object::Union{Plsr, Splsr}, nlv::Int)
    nlv = min(nlv, object.par.nlv)
    theta = vcol(object.C, 1:nlv)'  
    Dy = Diagonal(object.yscales)
    B = fweightr(vcol(object.R, 1:nlv), 1 ./ object.xscales) * theta * Dy
    int = object.ymeans' .- object.xmeans' * B
    (B = B, int)
end

"""
    predict(object::Union{Plsr, Splsr}, X)
    predict(object::Union{Plsr, Splsr}, X, nlv::Union{Int, AbstractVector{Int}})
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Union{Plsr, Splsr}, X)
    X = ensure_mat(X)
    coefs = coef(object)
    pred = coefs.int .+ X * coefs.B  # try muladd(X, coefs.B, coefs.int) 
    (pred = pred, nlv = object.par.nlv)
end

function predict(object::Union{Plsr, Splsr}, X, nlv::Union{Int, AbstractVector{Int}})
    X = ensure_mat(X)
    Q = eltype(X)
    a = object.par.nlv
    if isa(nlv, Int)
        nlv = min(nlv, a)
    else
        nlv = min(minimum(nlv), a):min(maximum(nlv), a)
    end
    le_nlv = length(nlv)
    pred = list(Matrix{Q}, le_nlv)
    @inbounds for i in eachindex(nlv)
        coefs = coef(object, nlv[i])
        pred[i] = coefs.int .+ X * coefs.B  
    end 
    (pred = pred, nlv)
end

#function predict(object::Union{Plsr, Splsr}, X)
#    X = ensure_mat(X)
#    Q = eltype(X)
#    nlv = object.par.nlv
#    res = predict(object, X, nlv)
#    (pred = res.pred[1], nlv)
#end





function predict2(object::Union{Plsr, Splsr}, X; nlv::Union{Nothing, Int, AbstractVector{Int}} = nothing)
    X = ensure_mat(X)
    Q = eltype(X)
    a = object.par.nlv
    if isnothing(nlv)
        nlv = a
    elseif isa(nlv, Int)
        nlv = min(nlv, a)
    else
        nlv = min(minimum(nlv), a):min(maximum(nlv), a)
    end
    le_nlv = length(nlv)
    pred = list(Matrix{Q}, le_nlv)
    @inbounds for i in eachindex(nlv)
        coefs = coef(object; nlv = nlv[i])
        pred[i] = coefs.int .+ X * coefs.B  # try muladd(X, coefs.B, coefs.int)
    end 
    if le_nlv == 1 ; pred = pred[1] ; end
    (pred = pred, nlv)
end


"""
    summary(object::Union{Plsr, Splsr}, X)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Union{Plsr, Splsr}, X)
    X = ensure_mat(X)
    n, nlv = size(object.T)
    X = fcscale(X, object.xmeans, object.xscales)
    ## Could be fcscale! but changes X. If too heavy ==> Makes summary!
    sstot = frob2(X, object.weights)       # = sum(object.weights.values' * X.^2)
    tt = object.TT
    tt_adj = (colnorm(object.V).^2) .* tt  # tt_adj[a] = p[a]'p[a] * tt[a]
    xvar = tt_adj / n    
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(nlv = collect(1:nlv), var = xvar, pvar = pvar, cumpvar = cumpvar)     
    (explvarx = explvarx,)
end
