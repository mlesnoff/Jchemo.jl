residreg(pred, Y) = Y - pred

residcla(pred, y) = pred .!= y

function msep(pred, Y)
    r = residreg(pred, Y)
    reshape(colmeans(r.^2), 1, :)
end

rmsep(pred, Y) = sqrt.(msep(pred, Y))

function bias(pred, Y)
    r = residreg(pred, Y)
    reshape(colmeans(r), 1, :)
end

sep(pred, Y) = sqrt.(msep(pred, Y) .- bias(pred, Y).^2)

function err(pred, y)
    r = residcla(pred, y)
    sum(r) / size(y, 1)
end

## Temporary version to be modified
function mse(pred, y)
    zmsep = msep(pred, y)
    zsep = sep(pred, y)
    zbias = bias(pred, y)
    (msep = zmsep, sep = zsep, bias = zbias)
end


