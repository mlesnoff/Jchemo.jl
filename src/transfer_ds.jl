struct TransferDs
    fm
end


"""
    transfer_ds(X1, X2; fun = mlrpinv, kwargs...)
Calibration transfer of spectral data with direct standardization (DS).
* `X1` : Target (standart) X-data (n, p).
* `X2` : X-data (n, p) to transfer to the standart.
* `fun` : Function used for fitting the transfer model.  
* `kwargs` : Optional arguments for `fun`.

`X1` and `X2` are assumed to represent the same n samples. 

## References

Y. Wang, D. J. Veltkamp, and B. R. Kowalski, “Multivariate Instrument Standardization,” 
Anal. Chem., vol. 63, no. 23, pp. 2750–2756, 1991, doi: 10.1021/ac00023a016.
""" 
function transfer_ds(X1, X2; fun = mlrpinv, kwargs...)
    fm = fun(X2, X1; kwargs...)
    TransferDs(fm)
end

"""
    predict(object::TransferDs, X; kwargs...)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `kwargs` : Optional arguments.
""" 
function predict(object::TransferDs, X; kwargs...)
    predict(object.fm, X; kwargs...)
end


