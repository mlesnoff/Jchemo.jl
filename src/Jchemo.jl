module Jchemo  # Start-Module

using Clustering
using DataInterpolations
using DecisionTree
using Distributions
using DataFrames
using Distances
using ImageFiltering     # convolutions in preprocessing (mavg, savgol)
using Interpolations
using LinearAlgebra
using Makie
using NearestNeighbors
using Random
using SparseArrays 
using Statistics
using StatsBase          # sample

include("utility.jl") 
include("colmedspa.jl")
include("fweight.jl") 
include("ellipse.jl")
include("matW.jl")
include("nipals.jl")
include("plotgrid.jl")
include("plotsp.jl")
include("plotxy.jl")
include("preprocessing.jl") 
include("rmgap.jl")

# Distributions
include("dmnorm.jl")
include("dmnormlog.jl")
include("dmkern.jl")

# Exploratory
include("fda.jl")     # Here since ::Fda called in pcasvd
include("fdasvd.jl")     
include("pcasvd.jl")
include("pcaeigen.jl")
include("kpca.jl")
include("rp.jl")
include("pcasph.jl") 

# Exploratory - Multiblock 
include("angles.jl")
include("mblock.jl")
include("blockscal.jl")
include("mbpca.jl")
include("comdim.jl")
include("mbunif.jl")
include("cca.jl")
include("ccawold.jl")
include("plscan.jl")
include("plstuck.jl")
include("rasvd.jl")

# Regression 
include("aov1.jl")
include("mlr.jl")
include("rr.jl")
include("pcr.jl")
include("rrr.jl") 
include("plskern.jl") ; include("plsrosa.jl")
include("plsnipals.jl") ; include("plssimp.jl")
include("plswold.jl") 
include("plsravg.jl")
include("plsravg_aic.jl")
include("plsravg_cv.jl")
include("plsravg_unif.jl")
include("plsravg_shenk.jl")
include("plsrstack.jl")
include("cglsr.jl")
include("covselr.jl")  
include("krr.jl")
include("kplsr.jl") ; include("dkplsr.jl")
include("aicplsr.jl")
include("wshenk.jl") 
include("vip.jl") 

include("xfit.jl")

# Regression Multiblock
include("mbplsr.jl") 
include("mbplswest.jl")
include("mbwcov.jl")
include("rosaplsr.jl") 
include("soplsr.jl") 

# Local regression
include("locw.jl")
include("locwlv.jl")
include("knnr.jl")
include("lwmlr.jl")
include("lwmlr_s.jl")
include("lwplsr.jl")
include("lwplsravg.jl")
include("lwplsr_s.jl")

# Bagging
include("baggr.jl")
include("baggr_util.jl")

# Trees
include("treer_dt.jl")

# Discrimination 
include("lda.jl")
include("qda.jl")
include("rda.jl")
include("kdeda.jl")
include("mlrda.jl")
include("rrda.jl")
include("plsrda.jl") 
include("plslda.jl") ; include("plsqda.jl")
include("plskdeda.jl")
include("plsrdaavg.jl") ; include("plsldaavg.jl") ; include("plsqdaavg.jl") 
include("krrda.jl")
include("kplsrda.jl") ; include("dkplsrda.jl")

include("occsd.jl") ; include("occod.jl") ; ; include("occsdod.jl")
include("occstah.jl") ; include("stah.jl")
include("occknndis.jl") ; include("occlknndis.jl")

include("cplsravg.jl")  # Here since call ::PlsrDa

# Trees 
include("treeda_dt.jl")

# Local discrimination
include("lwmlrda.jl") ; include("lwmlrda_s.jl")
include("lwplsrda.jl") ; include("lwplsrda_s.jl")
include("lwplslda.jl")
include("lwplsqda.jl")
include("lwplsrdaavg.jl")
include("lwplsldaavg.jl")
include("lwplsqdaavg.jl")
include("knnda.jl")

# Variable importance (direct methods) 
include("covsel.jl")
include("isel.jl")
include("viperm.jl")

# Validation
include("mpar.jl")
include("scores.jl")
include("confusion.jl")
include("gridscore.jl")
include("segm.jl")
include("gridcv.jl")
include("gridcv_mb.jl")
include("selwold.jl")

# Transfer
include("calds.jl")
include("calpds.jl")
include("difmean.jl")
include("eposvd.jl")

# Sampling
include("mtest.jl")
include("sampling.jl")

include("distances.jl")
include("getknn.jl")
include("wdist.jl")
include("kernels.jl")

export 
    # Utilities
    aggstat,
    dupl, miss,
    center, center!, 
    colmad, colmean, colnorm, colstd, colsum, colvar,
    corm, covm,
    cosm, cosv,
    cscale, cscale!, 
    dummy,
    ensure_df, ensure_mat,
    findmax_cla, 
    frob,
    fweight,
    head,
    list, 
    matB, matW, 
    mblock,
    mlev,
    mweight, mweight!,
    nco,
    normw,  
    nro,
    out,
    pmod, pnames, psize,
    pval,
    recodcat2int, recodnum2cla, 
    replacebylev, replacebylev2, 
    replacedict, 
    rmcol, rmrow, 
    rowmean, rowstd, rowsum,   
    scale, scale!,
    soft,
    sourcedir,
    ssq,
    summ,
    tab, tabdf, tabdupl,
    vcatdf,
    vcol, vrow,
    # Distributions
    dmnorm, dmnorm!,
    dmnormlog, dmnormlog!,
    dmkern,
    # Pre-processing
    detrend, detrend!, 
    fdif, fdif!,
    interpl, interpl_mon, 
    linear_int, quadratic_int, 
    quadratic_spline, cubic_spline,
    mavg, mavg!, 
    mavg_runmean, mavg_runmean!,
    rmgap, rmgap!,
    savgk, savgol, savgol!,
    snv, snv!, 
    # Transfer
    calds, calpds,
    difmean,
    eposvd,
    # Exploratory
    kpca,
    nipals,
    pcasvd, pcasvd!, 
    pcaeigen, pcaeigen!, pcaeigenk, pcaeigenk!,
    rpmatgauss, rpmatli, rp, rp!,
    pcasph, pcasph!,
    # Exploratory Multiblock
    blockscal, blockscal_frob, blockscal_mfa,
    blockscal_ncol, blockscal_sd,
    rv, lg, rd, 
    mbpca, mbpca!,
    comdim, comdim!,
    mbunif, mbunif!,  
    cca, cca!,
    ccawold, ccawold!,
    plscan, plscan!,
    plstuck, plstuck!,
    rasvd, rasvd!,
    # Regression
    aov1,
    mlr, mlr!, mlrchol, mlrchol!, 
    mlrpinv, mlrpinv!, mlrpinvn, mlrpinvn!,
    mlrvec, mlrvec!,
    plskern, plskern!, 
    plsnipals, plsnipals!, 
    plsrosa, plsrosa!, 
    plssimp, plssimp!,
    plswold, plswold!,
    cglsr, cglsr!,
    pcr,
    covselr,
    rr, rr!, rrchol, rrchol!,
    rrr, rrr!,   
    krr, krr!, kplsr, kplsr!, 
    dkplsr, dkplsr!,
    plsravg, plsravg!,
    dfplsr_cg, aicplsr,
    wshenk,
    treer_dt, rfr_dt, 
    baggr, 
    oob_baggr, vi_baggr, 
    # Rgression Multi-block
    mbplsr, mbplsr!,
    mbplswest, mbplswest!,
    mbwcov!, mbwcov,
    rosaplsr, rosaplsr!,
    soplsr,
    # Variable selection/importance (direct methods) 
    covsel,
    isel,
    vip, viperm,
    # Utils
    xfit, xfit!, xresid, xresid!,
    # Local regression
    locw, locwlv,
    knnr,
    lwmlr, lwmlr_s,
    lwplsr, lwplsravg, lwplsr_s,  
    cplsravg,
    # Discrimination
    fda, fda!, fdasvd, fdasvd!,
    mlrda,
    rrda, krrda,
    lda, qda, kdeda,
    rda,
    plsrda, kplsrda, dkplsrda,
    plslda, plsqda, plskdeda,
    plsrdaavg, plsldaavg, plsqdaavg,
    treeda_dt, rfda_dt,
    occsd, occod, occsdod,
    occstah, stah,
    occknndis, occlknndis,
    # Local Discrimination
    lwmlrda, lwmlrda_s,
    lwplsrda, lwplsrda_s,
    lwplslda, lwplsqda,
    lwplsrdaavg, lwplsldaavg, lwplsqdaavg,
    knnda,
    #
    transform, coef, predict,
    # Validation
    residreg, residcla, 
    ssr, msep, rmsep, rmsepstand, bias, sep, cor2, r2, rpd, rpdr, mse, err,
    mpar,
    gridscore, gridscorelv, gridscorelb,
    segmts, segmkf,
    gridcv, gridcvlv, gridcvlb, 
    gridcv_mb, gridcvlv_mb,
    selwold,
    confusion, 
    # Sampling
    mtest,
    sampks, sampdp, sampsys, sampcla, 
    # Distances
    getknn, wdist, wdist!,
    euclsq, mahsq, mahsqchol,
    krbf, kpol,
    # Graphics
    plotconf,
    plotgrid, 
    plotsp,
    plotxy
    # Not exported since surcharge:
    # - summary => Base.summary

end # End-Module




