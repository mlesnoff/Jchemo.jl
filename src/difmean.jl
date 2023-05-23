# """
#    bop(X1, X2; nlv)
# Build by Monte Carlo a detrimental matrix for calibration transfer without standard samples.
# * `X1` : Target spectra, (n1, p).
# * `X2` : Spectra to transfer to the target, (n, p).
# * `nlv` : Function used for fitting the transfer model.  
#
# The function returns a matrix D containing the detrimental information
# that has to be removed from spectra `X1` and `X2` by orthogonalization, 
# for instance by method EPO (see `?eposvd`). Matrix D is computed 
# by Monte Carlo sampling.
#
# ## References
#
# ## Examples
# ```julia
# ```

## Difference beween mean spectra 
function difmean(X1, X2)
    xmeans1 = colmean(X1)
    xmeans2 = colmean(X2)
    #xmeans1 ./= norm(xmeans1)
    #xmeans2 ./= norm(xmeans2)
    D = (xmeans1 - xmeans2)'
    (D = D, xmeans1, xmeans2)
end
