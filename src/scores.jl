"""
    residreg(pred, Y) 
Compute regression prediction errors.
* `pred` : Predictions.
* `Y` : Observed data.
"""
residreg(pred, Y) = Y - pred

"""
    residcla(pred, y) 
Compute classification prediction error (0 = no error, 1 = error).
* `pred` : Predictions.
* `Y` : Observed data.
"""
residcla(pred, y) = pred .!= y

"""
    ssr(pred, Y)
Compute the sum of squared prediction errors (SSR).
* `pred` : Predictions.
* `Y` : Observed data.
"""
function ssr(pred, Y)
    r = residreg(pred, Y)
    reshape(sum(r.^2, dims = 1), 1, :)
end

"""
    msep(pred, Y)
Compute the mean of the squared prediction errors (MSEP).
* `pred` : Predictions.
* `Y` : Observed data.
"""
function msep(pred, Y)
    r = residreg(pred, Y)
    reshape(colmeans(r.^2), 1, :)
end

"""
    rmsep(pred, Y)
Compute the square root of the mean of the squared prediction errors (RMSEP).
* `pred` : Predictions.
* `Y` : Observed data.
"""
rmsep(pred, Y) = sqrt.(msep(pred, Y))

"""
    bias(pred, Y)
Compute the prediction bias, i.e. the opposite of the mean prediction error.
* `pred` : Predictions.
* `Y` : Observed data.
"""
function bias(pred, Y)
    r = residreg(pred, Y)
    reshape(-colmeans(r), 1, :)
end

"""
    sep(pred, Y)
Compute the corrected SEP ("SEP_c"), i.e. the standard deviation of the prediction errors.
* `pred` : Predictions.
* `Y` : Observed data.

## References

Bellon-Maurel, V., Fernandez-Ahumada, E., Palagos, B., Roger, J.-M., McBratney, A., 2010.
Critical review of chemometric indicators commonly used for assessing the quality of 
the prediction of soil attributes by NIR spectroscopy. 
TrAC Trends in Analytical Chemistry 29, 1073–1081. https://doi.org/10.1016/j.trac.2010.05.006
"""
sep(pred, Y) = sqrt.(msep(pred, Y) .- bias(pred, Y).^2)

"""
    r2(pred, Y)
Compute the R2 coefficient.
* `pred` : Predictions.
* `Y` : Observed data.

The rate R2 is calculated by R2 = 1 - MSEP(current model) / MSEP(null model), 
where the "null model" is the overall mean. 
For predictions over CV or test sets, and/or for non linear models, 
it can be different from the square of the correlation coefficient (`cor2`) 
between the true data and the predictions. 
"""
function r2(pred, Y)
    m = size(Y, 1)
    mu = colmeans(Y)
    zmu = reduce(hcat, fill(mu, m, 1))'
    1 .- msep(pred, Y) ./ msep(zmu, Y)
end

"""
    cor2(pred, Y)
Compute the squared linear correlation between data and predictions.
* `pred` : Predictions.
* `Y` : Observed data.
"""
function cor2(pred, Y)
    q = size(Y, 2)
    res = cor(pred, Y).^2
    q == 1 ? nothing : res = diag(res)
    res = reshape(res, 1, :)
end

"""
    rpd(pred, Y)
Compute the ratio "deviation to model performance" (RPD).
* `pred` : Predictions.
* `Y` : Observed data.

This is the ratio of the deviation to the model performance to the deviation,
defined by RPD = Std(Y) / RMSEP, where Std(Y) is the standard deviation. 

Since Std(Y) = RMSEP(null model) where the null model is the simple average,
this also gives RPD = RMSEP(null model) / RMSEP. 
"""
rpd(pred, Y) = std(Y, dims = 1, corrected = false) ./ rmsep(pred, Y) 

"""
    rpdr(pred, Y)
Compute a robustified RPD.
* `pred` : Predictions.
* `Y` : Observed data.
"""
function rpdr(pred, Y)
    #Y = ensure_mat(Y)
    u = mapslices(Jchemo.mad, Y; dims = 1) / 1.4826
    r = residreg(pred, Y)
    v = mapslices(median, abs.(r); dims = 1)
    u ./ v
end


# Robust RPD = MAD(Y) / Median(Abs(prediction errors))
# MAD = Median(Abs(prediction errors)) where the median is the null model
#rpdr <- function(pred, Y) {
#    u <- apply(.mat(Y), FUN = mad, MARGIN = 2) / 1.4826
#    r <- residreg(pred, Y)
#    v <- apply(.mat(abs(r)), FUN = median, MARGIN = 2)
#    u / v
#}

"""
    mse(pred, Y)
Summary of model performance for regression.
* `pred` : Predictions.
* `Y` : Observed data.
"""
# Summary for regression
function mse(pred, Y)
    q = size(Y, 2)
    zmsep = msep(pred, Y)
    zrmsep = sqrt.(zmsep)
    zsep = sep(pred, Y)
    zbias = bias(pred, Y)
    zcor2 = cor2(pred, Y)    
    zr2 = r2(pred, Y)
    zrpd = rpd(pred, Y)
    zrpdr = rpdr(pred, Y)
    zmean = reshape(colmeans(Y), 1, :)
    nam = map(string, repeat(["y"], q), 1:q)
    nam = reshape(nam, 1, :)
    res = (nam = nam, msep = zmsep, rmsep = zrmsep, sep = zsep, bias = zbias, 
        cor2 = zcor2, r2 = zr2, rpd = zrpd, rpdr = zrpdr, mean = zmean)
    DataFrame(res)
end

"""
    err(pred, y)
Compute the classification error rate (ERR).
* `pred` : Predictions.
* `y` : Observed data.
"""
function err(pred, y)
    r = residcla(pred, y)
    sum(r) / size(y, 1)
end




