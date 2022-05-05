"""
    wshenk(object::Union{Pcr, Plsr}, X; nlv = nothing)
Compute the Shenk et al. (1997) "LOCAL" PLSR weights
* `object` : The fitted model.
* `X` : X-data on which the weights are computed.
* `nlv` : Nb. latent variables (LVs) to consider. If nothing, 
    it is the maximum nb. of components.

For each observation (row) of `X`, the weights are returned 
for the models with 1, ..., nlv LVs. 

## References
Shenk, J., Westerhaus, M., Berzaghi, P., 1997. Investigation of a LOCAL calibration 
procedure for near infrared instruments. 
Journal of Near Infrared Spectroscopy 5, 223. https://doi.org/10.1255/jnirs.115

Shenk et al. 1998 United States Patent (19). Patent Number: 5,798.526.

Zhang, M.H., Xu, Q.S., Massart, D.L., 2004. Averaged and weighted average partial 
least squares. Analytica Chimica Acta 504, 279–289. https://doi.org/10.1016/j.aca.2003.10.056

## Examples 
```julia 
using JLD2, CairoMakie
mypath = dirname(dirname(pathof(Jchemo)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y 
year = dat.Y.year

s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = 30
fm = plskern(Xtrain, ytrain; nlv = nlv) ;
res = Jchemo.wshenk(fm, Xtest) ;
pnames(res) 
plotsp(res.w, 1:nlv).f
```
""" 
function wshenk(object::Union{Pcr, Plsr}, X; nlv = nothing)
    X = ensure_mat(X)
    m, p = size(X)
    q = size(object.C, 1)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    E = similar(X)
    rss_r = similar(X, m, nlv)
    rss_b = similar(X, nlv)
    B = similar(X, p + 1, q)
    w = similar(X, m, nlv)
    wr = copy(w)
    for j = 1:nlv
        E .= xresid(object, X, nlv = j)
        rss_r[:, j] .= sqrt.(rowsum(E.^2))
        res = coef(object; nlv = j)
        B[1, :] .= vec(res.int)
        B[2:end, :] .= res.B
        #B = copy(res.B)
        rss_b[j] = sqrt.(ssq(B) / q)
    end 
    wb = mweight(1 ./ rss_b)
    for i = 1:m
        wr[i, :] .= mweight(1 ./ vrow(rss_r, i))
        w[i, :] = mweight(vrow(wr, i) .* wb) 
    end
    (w = w, wr = wr, wb = wb)
end

