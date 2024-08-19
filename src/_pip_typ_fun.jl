######## For fit!(::Pipeline)
######## Only functions that potentially enters in a pipeline 

struct FunX{FUN <: Function}
    fun::FUN   
end

struct FunXY{FUN <: Function}
    fun::FUN   
end

dt() = FunX{Function}(dt)
asls() = FunX{Function}(asls)
fdif() = FunX{Function}(fdif)
interpl() = FunX{Function}(interpl)
mavg() = FunX{Function}(mavg)
savgol() = FunX{Function}(savgol)
snorm() = FunX{Function}(snorm)
snv() = FunX{Function}(snv)
center() = FunX{Function}(center)
scale() = FunX{Function}(scale)
cscale() = FunX{Function}(cscale)
blockscal() = FunX{Function}(blockscal)
##
pcasvd() = FunX{Function}(pcasvd)
pcaeigen() = FunX{Function}(pcaeigen)
pcaeigenk() = FunX{Function}(pcaeigenk)
pcanipals() = FunX{Function}(pcanipals)n
pcanipalsmiss() = FunX{Function}(pcanipalsmiss)
pcasph() = FunX{Function}(pcasph)
spca() = FunX{Function}(spca)
kpca() = FunX{Function}(kpca)
rp() = FunX{Function}(rp)
umap() = FunX{Function}(umap)
##
mbconcat() = FunX{Function}(mbconcat)
mbpca() = FunX{Function}(mbpca)
comdim() = FunX{Function}(comdim)
##
fda() = FunXY{Function}(fda)
fdasvd() = FunXY{Function}(fdasvd)
##
mlr() = FunXY{Function}(mlr)
mlrchol() = FunXY{Function}(mlrchol)
mlrpinv() = FunXY{Function}(mlrpinv)
mlrpinvn() = FunXY{Function}(mlrpinvn)
mlrvec() = FunXY{Function}(mlrvec)
##
plskern() = FunXY{Function}(plskern)
plsnipals() = FunXY{Function}(plsnipals)
plswold() = FunXY{Function}(plswold)
plsrosa() = FunXY{Function}(plsrosa)
plssimp() = FunXY{Function}(plssimp)
cglsr() = FunXY{Function}(cglsr)
pcr() = FunXY{Function}(pcr)
rrr() = FunXY{Function}(rrr)
splskern() = FunXY{Function}(splskern)
plsravg() = FunXY{Function}(plsravg)
kplsr() = FunXY{Function}(kplsr)
dkplsr() = FunXY{Function}(dkplsr)
rr() = FunXY{Function}(rr)
rrchol() = FunXY{Function}(rrchol)
krr() = FunXY{Function}(krr)
## 
knnr() = FunXY{Function}(knnr)
lwmlr() = FunXY{Function}(lwmlr)
lwplsr() = FunXY{Function}(lwplsr)
lwplsravg() = FunXY{Function}(lwplsravg)
loessr() = FunX{Function}(loessr)
##
svmr() = FunXY{Function}(svmr)
treer() = FunXY{Function}(treer)
rfr() = FunXY{Function}(rfr)
##
mbplsr() = FunXY{Function}(mbplsr)
mbplswest() = FunXY{Function}(mbplswest)
rosaplsr() = FunXY{Function}(rosaplsr)
soplsr() = FunXY{Function}(soplsr)
## 
mlrda() = FunXY{Function}(mlrda)
plsrda() = FunXY{Function}(plsrda)
rrda() = FunXY{Function}(rrda)
splsrda() = FunXY{Function}(splsrda)
kplsrda() = FunXY{Function}(kplsrda)
dkplsrda() = FunXY{Function}(dkplsrda)
krrda() = FunXY{Function}(krrda)
##
lda() = FunXY{Function}(lda)
qda() = FunXY{Function}(qda)
rda() = FunXY{Function}(rda)
kdeda() = FunXY{Function}(kdeda)
plslda() = FunXY{Function}(plslda)
plsqda() = FunXY{Function}(plsqda)
plskdeda() = FunXY{Function}(plskdeda)
kplslda() = FunXY{Function}(kplslda)
kplsqda() = FunXY{Function}(kplsqda)
kplskdeda() = FunXY{Function}(kplskdeda)
dkplslda() = FunXY{Function}(dkplslda)
dkplsqda() = FunXY{Function}(dkplsqda)
dkplskdeda() = FunXY{Function}(dkplskdeda)
##
splslda() = FunXY{Function}(splslda)
splsqda() = FunXY{Function}(splsqda)
splskdeda() = FunXY{Function}(splskdeda)
##
knnda() = FunXY{Function}(knnda)
lwmlrda() = FunXY{Function}(lwmlrda)
lwplsrda() = FunXY{Function}(lwplsrda)
lwplslda() = FunXY{Function}(lwplslda)
lwplsqda() = FunXY{Function}(lwplsqda)
##
svmda() = FunXY{Function}(svmda)
treeda() = FunXY{Function}(treeda)
rfda() = FunXY{Function}(rfda)
## 
mbplsrda() = FunXY{Function}(mbplsrda)
mbplslda() = FunXY{Function}(mbplslda)
mbplsqda() = FunXY{Function}(mbplsqda)
mbplskdeda() = FunXY{Function}(mbplskdeda)
