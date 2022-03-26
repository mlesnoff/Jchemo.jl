struct TransferPds2
    fm
    s
end


"""
    transfer_pds(X1, X2; fun = mlrpinv, m = 5, kwargs...)
Calibration transfer of spectral data with direct standardization methods.
* `X1` : Target (standart) X-data (n, p).
* `X2` : X-data (n, p) to transfer to the standart.
* `fun` : Function used for fitting the transfer model.  
* `kwargs` : Optional arguments for `fun`.

`X1` and `X2` are assumed to represent the same n samples. 

## References

Y. Wang, D. J. Veltkamp, and B. R. Kowalski, “Multivariate Instrument Standardization,” 
Anal. Chem., vol. 63, no. 23, pp. 2750–2756, 1991, doi: 10.1021/ac00023a016.
    
""" 
function transfer_pds(X1, X2; fun = mlrpinv, m = 5, kwargs...)
    p = nco(X1)
    fm = list(p)
    s = list(p)
    zm = repeat([m], p)
    zm[1:m] .= collect(1:m) .- 1
    zm[(p - m + 1):p] .= collect(m:-1:1) .- 1
    @inbounds for i = 1:p
        s[i] = collect((i - zm[i]):(i + zm[i]))
        fm[i] = fun(X2[:, s[i]], X1[:, i]; kwargs...)
    end
    TransferPds2(fm, s)
end

"""
    predict(object::TransferDs1, X; kwargs...)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `kwargs` : Optional arguments.
""" 
function predict(object::TransferPds2, X; kwargs...)
    m, p = size(X)
    pred = similar(X, m, p)
    for i = 1:p 
        pred[:, i] .= predict(object.fm[i], X[:, object.s[i]]; kwargs...).pred
    end
    (pred = pred,)
end



