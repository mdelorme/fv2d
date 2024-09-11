#pragma once

namespace fv2d {

const __device__ real_t rho_coef[] = {0x1.00000954ecccdp+0, 0x1.8313a1c0e2823p-9, -0x1.63f6d861c4dafp+3, -0x1.6c95fa6f7ab58p+2, 0x1.2315a281c904ap+7, -0x1.345b589764c4cp+10, 0x1.3725cb1ceeef5p+13, -0x1.7bc7062b82d9fp+14, -0x1.158e5ec8a0e1bp+17, 0x1.2be5d8c562c9bp+20, -0x1.f0774b2f7c8e8p+21, 0x1.abc8445e85170p+22, -0x1.2dbc40aa433eep+22, -0x1.7113d98b9138dp+21, 0x1.89faabcb189eep+22, 0x1.5efa0586d16fap+20, -0x1.873ed23f0c6eap+22, -0x1.03e7df91a77efp+21, 0x1.6273ec296b680p+22, 0x1.e9c1b62370eccp+21, -0x1.fc965b2127e82p+21, -0x1.67d463950708dp+22, 0x1.4ca6a9714a554p+20, 0x1.98f32ba5ba27ep+22, 0x1.bb870a047eafdp+20, -0x1.7e5b9e14affcap+22, -0x1.0349ef3f9358bp+22, 0x1.6c9ce9cb8f362p+22, 0x1.2fa5f9ea9b731p+22, -0x1.1e3cb7329accap+23, 0x1.219a6cf31d422p+22, -0x1.98d481a9c6241p+19 };
const __device__ real_t prs_coef[] = {0x1.0000199d7999ap+0, -0x1.ec9f305744494p-9, -0x1.2457963034680p+4, -0x1.749fb404c8153p+4, 0x1.579750b07381dp+9, -0x1.f90f380c74e87p+12, 0x1.32298206c6347p+16, -0x1.c59c63a5022e9p+18, 0x1.94ce50b8ccd0dp+20, -0x1.ad925765b783dp+21, 0x1.c8daef472ca7ap+21, 0x1.44aae7db62b0ap+18, -0x1.3c834a9a933f3p+22, 0x1.301acf317d3a2p+21, 0x1.2bc5b8d583f5ap+22, -0x1.5f7612e102ba3p+21, -0x1.4639361ef23f6p+22, 0x1.8b58cd937408cp+20, 0x1.6ba96cdeaedbdp+22, 0x1.c811ec91e6764p+19, -0x1.53d6692af8c08p+22, -0x1.e1162e94c0017p+21, 0x1.a8f181ca54097p+21, 0x1.7230165b1e857p+22, -0x1.0504310db93d9p+18, -0x1.9299290514abap+22, -0x1.403469569866ep+21, 0x1.90b524183b014p+22, 0x1.c389a16459605p+21, -0x1.1ade11f808133p+23, 0x1.353de47252625p+22, -0x1.c9d1241b74f32p+19 };
const __device__ real_t g_coef[] = {0x1.94ff333333333p-16, -0x1.2c92ca0962cdfp+5, 0x1.4599851f5e7c1p+3, -0x1.112bba2fe9e92p+9, 0x1.1254c978893aep+14, -0x1.ebcc91932cba3p+17, 0x1.4444d62e8ce5fp+21, -0x1.1c2514bf38115p+24, 0x1.45d09647b0e43p+26, -0x1.ee143f0e1237cp+27, 0x1.e5d1784ca47c3p+28, -0x1.157a09d73d1a5p+29, 0x1.766b3e3361d52p+27, 0x1.5b78c6b2d0120p+28, -0x1.41b2bc0391857p+28, -0x1.d4cdf16c98c95p+27, 0x1.2cbe30f025deep+28, 0x1.dd6d318c2e5ffp+27, -0x1.c63e1245bf1a6p+27, -0x1.1ef627bae4aedp+28, 0x1.b90f819cb7c45p+26, 0x1.3ce10aae84185p+28, 0x1.fb3de7c79fc77p+24, -0x1.2777fd5e1bf2dp+28, -0x1.3c6366710eef7p+27, 0x1.dd06c5afcdda0p+27, 0x1.daa3c1a97d248p+27, -0x1.b06fe9833dd04p+27, -0x1.f8a9cd5403144p+27, 0x1.7defac821ad76p+28, -0x1.638d04f8bdb78p+27, 0x1.d96bbda8d1585p+24 };
const __device__ real_t dTdr_coef[] = {-0x1.f86e60ccccccdp-8, -0x1.c8c1801b3aa35p+3, -0x1.70253fabe726ep+5, 0x1.60877ab5ff677p+10, -0x1.746f435be26c8p+14, 0x1.0bb743d918644p+18, -0x1.ee64289b8c07ap+20, 0x1.3c200bd570546p+23, -0x1.27a84f5de09ccp+25, 0x1.98f5936d5f3d5p+26, -0x1.93a501af48122p+27, 0x1.fcd0fa4fa33bap+27, -0x1.0541dc8bad2abp+27, -0x1.fa05a7ee8d7d5p+26, 0x1.82bff009077fcp+27, 0x1.03f93deecbb4ap+26, -0x1.847695fe6e2d5p+27, -0x1.34e575ef46507p+26, 0x1.615094f9e2ffdp+27, 0x1.044627cf5bbc3p+27, -0x1.00906cd124202p+27, -0x1.7746765fe30bcp+27, 0x1.5817fb7bb9013p+25, 0x1.ae8d8491542b8p+27, 0x1.c9f1e75d25bfdp+25, -0x1.9a0513d50e4f5p+27, -0x1.1203fc4500f01p+27, 0x1.8dd37497a550fp+27, 0x1.451666fcea9e4p+27, -0x1.39b13fa8f4160p+28, 0x1.41563d476262bp+27, -0x1.ca7dad632c20ep+24 };

KOKKOS_INLINE_FUNCTION
real_t poly_eval(real_t r, const real_t *coef, size_t len){
  real_t ret = 0; 
  real_t poly_r = 1.;
  for(size_t i=0; i<len; i++){
    ret += poly_r * coef[i];
    poly_r *= r;
  }
  return ret;
}

#define get_rho(x)   poly_eval(x,   rho_coef, sizeof(  rho_coef)/sizeof(  *rho_coef))
#define get_prs(x)   poly_eval(x,   prs_coef, sizeof(  prs_coef)/sizeof(  *prs_coef))
#define get_g(x)     poly_eval(x,     g_coef, sizeof(    g_coef)/sizeof(    *g_coef))
// #define get_kappa(x) poly_eval(x, kappa_coef, sizeof(kappa_coef)/sizeof(*kappa_coef))
#define get_kappa(x) (-1./poly_eval(x, dTdr_coef, sizeof(dTdr_coef)/sizeof(*dTdr_coef)))
}
