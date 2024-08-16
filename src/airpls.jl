"""
    airpls(X; kwargs...)
Baseline correction of each row of X-data by adaptive iteratively 
    reweighted penalized least squares algorithm (AIRPLS).
* `X` : X-data (n, p).
Keyword arguments:
* `lb` : Penalizing (smoothing) parameter "lambda".
* `maxit` : Maximum number of iterations.
* `verbose` : If `true`, nb. iterations are printed.

See Zhang et al. 2010, and Baek et al. 2015 section 2.

## References

Baek, S.-J., Park, A., Ahn, Y.-J., Choo, J., 2015. Baseline correction using 
asymmetrically reweighted penalized least squares smoothing. Analyst 140, 250–257. 
https://doi.org/10.1039/C4AN01061B

Zhang, Z.-M., Chen, S., Liang, Y.-Z., 2010. Baseline correction using adaptive 
iteratively reweighted penalized least squares. Analyst 135, 1138–1146. 
https://doi.org/10.1039/B922045C

https://github.com/zmzhang/airPLS/tree/master 

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

## Example on 1 spectrum
i = 2
zX = Matrix(X)[i:i, :]
lb = 1e6
mod = model(airpls; lb)
fit!(mod, zX)
zXc = transf(mod, zX)   # = corrected spectrum 
B = zX - zXc            # = estimated baseline
f, ax = plotsp(zX, wl)
lines!(wl, vec(B); color = :blue)
lines!(wl, vec(zXc); color = :black)
f
```
""" 
function airpls(X; kwargs...)
    par = recovkw(ParAirpls, kwargs).par
    Airpls(par)
end

""" 
    transf(object::Airpls, X)
    transf!(object::Airpls, X)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Airpls, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Airpls, X::Matrix)
    n, p = size(X)
    w = ones(p) 
    z = similar(X, p)
    z0 = copy(z)
    W = similar(X, p, p)
    D1 = sparse(diff(I(p); dims = 1))
    D2 = diff(D1; dims = 1)
    C = D2' * D2
    lb = object.par.lb
    maxit = object.par.maxit
    verbose = object.par.verbose 
    verbose ? println("Nb. iterations:") : nothing
    @inbounds for i = 1:n
        iter = 1
        cont = true
        x = vrow(X, i)
        normx = sum(abs.(x))  # alternative: norm(x)
        while cont
            z0 .= copy(z)
            W .= spdiagm(0 => w)  
            z .= cholesky!(Hermitian(W + lb * C)) \ (w .* x)
            rho = sum(-(x - z) .* (x .< z))  # alternative: norm((x - z) .* (x .< z))
            w .= (exp.(iter * (x - z) / rho)) .* (x .< z)  
            iter = iter + 1
            if (rho < .001 * normx) || (iter > maxit)
                cont = false
            end
        end
        verbose ? print(iter - 1, " ") : nothing
        X[i, :] .= x .- z
    end
end


