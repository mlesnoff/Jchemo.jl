"""
    pcoutpcout(X; explvar = .99, critm1 = 1 / 3, critc1 = 2.5, critm2 = 1 / 4, critc2 = 0.99, 
        cs = .25, outbound = 0.25)
    pcout!pcout(X::Matrix; explvar = .99, critm1 = 1 / 3, critc1 = 2.5, critm2 = 1 / 4, critc2 = 0.99, 
        cs = .25, outbound = 0.25)
Pcout algorithm for outlier identification in high dimensions.
* `X` : X-data (n, p).
Keyword arguments:
* `explvar` : A numeric value between 0 and 1 indicating how much variance should be covered
    by the robust PCs (default to 0.99).
* `critm1` : A numeric value between 0 and 1 indicating the quantile to be used as lower boundary 
    for location outlier detection (default to 1 / 3).
* `critc1` : A positive numeric value used for determining the upper boundary for location outlier 
    detection (default to 2.5).
* `critm2` : A numeric value between 0 and 1 indicating the quantile to be used as lower boundary 
    for scatter outlier detection (default to .25).
* `critc2` : A numeric value between 0 and 1 indicating the quantile to be used as upper boundary 
    for scatter outlier detection (default to 0.99).
* `cs` : A numeric value indicating the scaling constant for combined location and scatter 
    weights (default to 0.25).
* `outbound` : A numeric value between 0 and 1 indicating the outlier boundary for defining values 
    as final outliers (default to 0.25).

This is the PCOut algorithm of Filzmoser et al. (2008). Same algorithm as the one implemented in the R 
package mvoutlier (Filzmoser, 2026).

The function returns a named tuple with the following outputs:
* `wfinal01` : 0/1 vector with final weights for each observation; weight 0 indicates potential multivariate outliers.
* `wfinal` : Final weights for each observation; small values indicate potential multivariate outliers.
* `wloc` : Weights for each observation; small values indicate potential location outliers.
* `wscat` : Weights for each observation; small values indicate potential scatter outliers.
* `dist1` : Distances for location outlier detection.
* `dist2` : Distances for scatter outlier detection.
* `M1` : Upper boundary for assigning weight 1 in location outlier detection.
* `const1` : Lower boundary for assigning weight 0 in location outlier detection.
* `M2` : Upper boundary for assigning weight 1 in scatter outlier detection.
* `const2` : Lower boundary for assigning weight 0 in scatter outlier detection.

## References

Filzmoser, P., Maronna, R., Werner, M., 2008. Outlier identification in high dimensions. 
Computational Statistics & Data Analysis 52, 1694–1711. https://doi.org/10.1016/j.csda.2007.05.018

Filzmoser, P. 2026. mvoutlier: Multivariate Outlier Detection Based on Robust Methods. 
R package version 2.1.4. https://cran.r-project.org/package=mvoutlier

## Examples
```julia
using Jchemo, JLD2, CairoMakie
using JchemoData 
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "octane.jld2")
@load db dat
X = dat.X
wlst = names(X)
wl = parse.(Float64, wlst)
n, p = size(X)
## Six of the samples (25, 26, and 36-39) contain added alcohol
s = [25; 26; 36:39]
typ = zeros(Int, n)
typ[s] .= 1
#plotsp(X, wl; xlabel = "Wavelength (nm)", ylabel = "Absorbance").f

res = pcout(X) ;
@names res  
res.wfinal 

f = Figure(size = (700, 500))
ax1 = Axis(f[1, 1]; xlabel = "Observation", ylabel = "Distance (location)")
scatter!(ax1, 1:n, res.dist1)
hlines!(ax1, res.M1; linewidth = .7)
hlines!(ax1, res.const1; linewidth = .7)
ax2 = Axis(f[2, 1]; xlabel = "Observation", ylabel = "Distance (scatter)")
scatter!(ax2, 1:n, res.dist2)
hlines!(ax2, res.M2; linewidth = .7)
hlines!(ax2, res.const2; linewidth = .7)
ax3 = Axis(f[3, 1]; xlabel = "Observation", ylabel = "Final weight")
scatter!(ax3, 1:n, res.wfinal)
ax4 = Axis(f[1, 2]; xlabel = "Observation", ylabel = "Weight (location)")
scatter!(ax4, 1:n, res.wloc)
ax5 = Axis(f[2, 2]; xlabel = "Observation", ylabel = "Weight (scatter)")
scatter!(ax5, 1:n, res.wscat)
ax6 = Axis(f[3, 2]; xlabel = "Observation", ylabel = "Final 0/1 weight")
scatter!(ax6, 1:n, res.wfinal01)
f
```
""" 
function pcout(X; explvar = .99, critm1 = 1 / 3, critc1 = 2.5, critm2 = 1 / 4, critc2 = 0.99, 
    cs = .25, outbound = 0.25)
    pcout!(copy(ensure_mat(X)); explvar, critm1, critc1, critm2, critc2, cs, outbound) 
end

function pcout!(X::Matrix; explvar = .99, critm1 = 1 / 3, critc1 = 2.5, critm2 = 1 / 4, critc2 = 0.99, 
    cs = .25, outbound = 0.25)
    Q = eltype(X)
    n = nro(X)
    d = similar(X, n)
    w1 = similar(d)
    w2 = similar(d)
    wfinal = similar(d)
    wfinal01 = similar(d)
    fcscale!(X, colmed(X), colmad(X))
    res = svd(fcenter(X, colmean(X))) ;
    sv = res.S.^2 / (n - 1)
    nlv = findall(cumsum(sv) / sum(sv) .> convert(Q, explvar))[1]
    distr = Chisq(nlv)    
    q = quantile(distr, .5)
    T = X * vcol(res.V, 1:nlv)             # PCs
    fcscale!(T, colmed(T), colmad(T))      # centered and scaled PCs (by median and mad)
    # Phase 1
    w = abs.(colmean(T.^4) .- 3)           # robust kurtosis for each PC
    w ./= sum(w)
    Tw = fweightc(T, w)                    # weighted PCs
    d .= sqrt.(rowsum(Tw.^2))              # weighted Mahalanobis distance in the PC space (RDi in Eq12)
    dist1 = d * sqrt(q) / median(d)        # di = transformed RDi
    M1 = quantile(dist1, convert(Q, critm1))  
    const1 = medv(dist1) + convert(Q, critc1) * madv(dist1)
    @inbounds for i in eachindex(w1)
        if dist1[i] <= M1
            w1[i] = 1
        elseif dist1[i] >= const1
            w1[i] = 0
        else
            w1[i] = (1 - ((dist1[i] - M1) / (const1 - M1))^2)^2
        end
    end 
    ## Phase 2
    d .= sqrt.(rowsum(T.^2))
    dist2 = d * sqrt(q) / median(d)
    M2 = sqrt(quantile(distr, convert(Q, critm2)))
    const2 = sqrt(quantile(distr, convert(Q, critc2)))   
    @inbounds for i in eachindex(w2)
        if dist2[i] <= M2
            w2[i] = 1
        elseif dist2[i] >= const2
            w2[i] = 0
        else
            w2[i] = (1 - ((dist2[i] - M2) / (const2 - M2))^2)^2
        end
    end 
    ## End
    c = convert(Q, cs)
    @. wfinal = ((w1 + c) * (w2 + c)) / (1 + c)^2
    @. wfinal01 = round(wfinal + 0.5 - convert(Q, outbound))   # weights < outbound are assigned 0
    (wfinal01 = wfinal01, wfinal, wloc = w1, wscat = w2, dist1, dist2, M1, const1, M2, const2)
end

