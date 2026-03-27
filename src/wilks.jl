"""
    wilks(A; digits = 5)
Compute statistics for multivariate tests.
* `A` : Matrix involved in the test.
Keyword arguments:
* `digits` : Nb. digits for the outputs.

Return
* Wilks’ lambda
* Pillai’s trace
* Hotelling-Lawley trace
* Roy’s maximum root

## References

https://documentation.sas.com/doc/en/statug/15.2/statug_introreg_sect038.htm#statug_introreg001918

## Examples 
```julia
```
"""
function wilks(A;  digits = 5)
    v = eigen(A; sortby = x -> -abs(x)).values
    wilks = cumprod(1 ./ (1 .+ v))[end]
    pillai = sum(v ./ (1 .+ v))
    hotel = sum(v)
    roy = v[1]
    res = round.((wilks, pillai, hotel, roy); digits)
    (; zip([:wilks, :pillai, :hotel, :roy], res)...)
end
