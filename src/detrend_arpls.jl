"""
    detrend_arpls(X; kwargs...)
Baseline correction of each row of X-data by asymmetrically
    reweighted penalized least squares smoothing (ARPLS).
* `X` : X-data (n, p).
Keyword arguments:
* `lb` : Penalizing (smoothness) parameter "lambda".
* `tol` : Tolerance value for stopping the iterations.  
* `maxit` : Maximum number of iterations.
* `verbose` : If `true`, nb. iterations are printed.

De-trend transformation: the function fits a baseline by ARPLS (see Baek et al. 2015 section 3)
for each observation and returns the residuals (= signals corrected from the baseline).

## References

Baek, S.-J., Park, A., Ahn, Y.-J., Choo, J., 2015. Baseline correction using 
asymmetrically reweighted penalized least squares smoothing. Analyst 140, 250–257. 
https://doi.org/10.1039/C4AN01061B

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
lb = 1e4
model = mod_(detrend_arpls; lb, p)
fit!(model, zX)
zXc = transf(model, zX)   # = corrected spectrum 
B = zX - zXc            # = estimated baseline
f, ax = plotsp(zX, wl)
lines!(wl, vec(B); color = :blue)
lines!(wl, vec(zXc); color = :black)
f
```
""" 
function detrend_arpls(X; kwargs...)
    par = recovkw(ParDetrendArpls, kwargs).par
    DetrendArpls(par)
end

""" 
    transf(object::DetrendArpls, X)
    transf!(object::DetrendArpls, X)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::DetrendArpls, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::DetrendArpls, X::Matrix)
    n, p = size(X)
    w = ones(p) 
    z = similar(X, p)
    z0 = copy(z)
    d = copy(z)
    W = similar(X, p, p)
    D1 = sparse(diff(I(p); dims = 1))
    D2 = diff(D1; dims = 1)
    C = D2' * D2
    lb = object.par.lb
    tol = object.par.tol
    maxit = object.par.maxit
    verbose = object.par.verbose 
    verbose ? println("Nb. iterations:") : nothing
    @inbounds for i = 1:n
        iter = 1
        cont = true
        x = vrow(X, i)
        while cont
            z0 .= copy(z)
            W .= spdiagm(0 => w)    
            z .= cholesky!(Hermitian(W + lb * C)) \ (w .* x)
            d .= (x - z)
            v = d[x .< z]
            mu = mean(v)
            sd = std(v)
            w .= 1 ./ (1 .+ exp.(2 * (d .- (-mu + 2 * sd)) / sd))
            dif = sum((z .- z0).^2)
            iter = iter + 1
            if (dif < tol) || (iter > maxit)
                cont = false
            end
        end
        verbose ? print(iter - 1, " ") : nothing
        X[i, :] .= x .- z
    end
end


