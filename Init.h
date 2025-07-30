#pragma once

#include <fstream>
#include <Kokkos_Random.hpp>

#include "SimInfo.h"
#include "BoundaryConditions.h"

namespace fv2d {

namespace {

  using RandomPool = Kokkos::Random_XorShift64_Pool<>;

  /**
   * @brief Sod Shock tube aligned along the X axis
   */
  KOKKOS_INLINE_FUNCTION
  void initSodX(Array Q, int i, int j, const DeviceParams &params) {
    if (getPos(params, i, j)[IX] <= 0.5) {
      Q(j, i, IR) = 1.0;
      Q(j, i, IP) = 1.0;
      Q(j, i, IU) = 0.0;
    }
    else {
      Q(j, i, IR) = 0.125;
      Q(j, i, IP) = 0.1;
      Q(j, i, IU) = 0.0;
    }
  }

  /**
   * @brief Sod Shock tube aligned along the Y axis
   */
  KOKKOS_INLINE_FUNCTION
  void initSodY(Array Q, int i, int j, const DeviceParams &params) {
    if (getPos(params, i, j)[IY] <= 0.5) {
      Q(j, i, IR) = 1.0;
      Q(j, i, IP) = 1.0;
      Q(j, i, IU) = 0.0;
    }
    else {
      Q(j, i, IR) = 0.125;
      Q(j, i, IP) = 0.1;
      Q(j, i, IU) = 0.0;
    }
  }

  
  /**
   * @brief Sedov blast initial conditions
   */
  KOKKOS_INLINE_FUNCTION
  void initBlast(Array Q, int i, int j, const DeviceParams &params) {
    real_t xmid = 0.5 * (params.xmin+params.xmax);
    real_t ymid = 0.5 * (params.ymin+params.ymax);
    
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];
    
    real_t xr = xmid - x;
    real_t yr = ymid - y;
    real_t r = sqrt(xr*xr+yr*yr);
    
    if (r < 0.2) {
      Q(j, i, IR) = 1.0;
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = 10.0;
    }
    else {
      Q(j, i, IR) = 1.2;
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = 0.1;
    }
  }


  /**
   * @brief Rising bubble initial conditions
   */
  KOKKOS_INLINE_FUNCTION
  void initHotBubble(Array Q, int i, int j, const DeviceParams &params) {
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    real_t xr = params.hot_bubble_x0 - x;
    real_t yr = params.hot_bubble_y0 - y;
    real_t r = sqrt(xr*xr+yr*yr);

    real_t ky = 2.0 * M_PI / params.ymax;
    real_t coeff_gamma = 1 - 1/params.gamma0;

    real_t prs = coeff_gamma * (1 - Kokkos::cos(ky * y)) * params.hot_bubble_g0 / (ky * pow(params.hot_bubble_A0, 1/params.gamma0));
    prs += pow(params.hot_bubble_p0, coeff_gamma);
    prs = pow(prs, 1/coeff_gamma);

    real_t coeff_rad = Kokkos::cos(M_PI / 2.0 * r / params.hot_bubble_r0);

    real_t A_inside = params.hot_bubble_A0 * (1 + params.hot_bubble_amplitude * coeff_rad * coeff_rad);

    if (r <= params.hot_bubble_r0 && params.hot_bubble_has_bubble) {
      Q(j, i, IR) = pow(prs/A_inside, 1/params.gamma0);
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = prs;
    }
    else {
      Q(j, i, IR) = pow(prs/params.hot_bubble_A0, 1/params.gamma0);
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = prs;
    }
  
  }
  

  /**
   * @brief Stratified convection based on Hurlburt et al 1984
   */
  KOKKOS_INLINE_FUNCTION
  void initH84(Array Q, int i, int j, const DeviceParams &params, const RandomPool &random_pool) {
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    real_t rho = pow(y, params.m1);
    real_t prs = pow(y, params.m1+1.0); 
    
    auto generator = random_pool.get_state();
    real_t pert = params.h84_pert * (generator.drand(-0.5, 0.5));
    random_pool.free_state(generator);
    
    Q(j, i, IR) = rho;
    Q(j, i, IU) = 0.0;
    Q(j, i, IV) = pert;
    Q(j, i, IP) = prs;
  }

  /**
   * @brief Stratified convection based on Cattaneo et al. 1991
   */
  KOKKOS_INLINE_FUNCTION
  void initC91(Array Q, int i, int j, const DeviceParams &params, const RandomPool &random_pool) {
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];
    
    real_t T = (1.0 + params.theta1*y);
    real_t rho = pow(T, params.m1);
    real_t prs = pow(T, params.m1+1.0);
    
    auto generator = random_pool.get_state();
    real_t pert = params.c91_pert * (generator.drand(-0.5, 0.5));
    random_pool.free_state(generator);
    
    prs = prs * (1.0 + pert);

    Q(j, i, IR) = rho;
    Q(j, i, IU) = 0.0;
    Q(j, i, IV) = 0.0;
    Q(j, i, IP) = prs;
  }

  /**
   * @brief Simple diffusion test with a structure being advected on the grid
   */
  KOKKOS_INLINE_FUNCTION
  void initDiffusion(Array Q, int i, int j, const DeviceParams &params) {
    real_t xmid = 0.5 * (params.xmin+params.xmax);
    real_t ymid = 0.5 * (params.ymin+params.ymax);

    Pos pos = getPos(params, i, j);
    
    real_t x0 = (pos[IX]-xmid);
    real_t y0 = (pos[IY]-ymid);
    
    real_t r = sqrt(x0*x0+y0*y0);
    
    if (r < 0.2) 
    Q(j, i, IR) = 1.0;
    else
    Q(j, i, IR) = 0.1;
    
    Q(j, i, IP) = 1.0;
    Q(j, i, IU) = 1.0;
    Q(j, i, IV) = 1.0;
  }
  
  /**
   * @brief Rayleigh-Taylor instability setup
   */
  KOKKOS_INLINE_FUNCTION
  void initRayleighTaylor(Array Q, int i, int j, const DeviceParams &params) {
    real_t ymid = 0.5*(params.ymin + params.ymax);
    
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    const real_t P0 = 2.5;
    
    if (y < ymid) {
      Q(j, i, IR) = 1.0;
      Q(j, i, IU) = 0.0;
      Q(j, i, IP) = P0 + 0.1 * params.gy * y;
    }
    else {
      Q(j, i, IR) = 2.0;
      Q(j, i, IU) = 0.0;
      Q(j, i, IP) = P0 + 0.1 * params.gy * y;
    }
    
    if (y > -1.0/3.0 && y < 1.0/3.0)
    Q(j, i, IV) = 0.01 * (1.0 + cos(4*M_PI*x)) * (1 + cos(3.0*M_PI*y))/4.0;
  }
  
  #ifdef MHD

  // 1D MHD Tests
  /**
   * @brief Brio-Wu (MHD Sod Shock) tube aligned along the X axis
   */
  KOKKOS_INLINE_FUNCTION
  void initMHDSodX(Array Q, int i, int j, const DeviceParams &params) {
    if (getPos(params, i, j)[IX] <= 0.5) {
      Q(j, i, IR) = 1.0;
      Q(j, i, IP) = 1.0;
      Q(j, i, IBY) = 1.0;
    }
    else {
      Q(j, i, IR) = 0.125;
      Q(j, i, IP) = 0.1;
      Q(j, i, IBY) = -1.0;
    }
    Q(j, i, IU) = 0.0;
    Q(j, i, IV) = 0.0;
    Q(j, i, IW) = 0.0;
    Q(j, i, IBX) = 0.75;
    Q(j, i, IBZ) = 0.0;
    Q(j, i, IPSI) = 0.0;
  }

    /**
   * @brief MHD Sod Shock tube aligned along the Y axis
   */
  KOKKOS_INLINE_FUNCTION
  void initMHDSodY(Array Q, int i, int j, const DeviceParams &params) {
    if (getPos(params, i, j)[IY] <= 0.5) {
      Q(j, i, IR) = 1.0;
      Q(j, i, IP) = 1.0;
      Q(j, i, IBX) = 1.0;
    }
    else {
      Q(j, i, IR) = 0.125;
      Q(j, i, IP) = 0.1;
      Q(j, i, IBX) = -1.0;
    }
    Q(j, i, IU) = 0.0;
    Q(j, i, IV) = 0.0;
    Q(j, i, IW) = 0.0;
    Q(j, i, IBY) = 0.75;
    Q(j, i, IBZ) = 0.0;
    Q(j, i, IPSI) = 0.0;
  }

  /**
   * @brief Dai and Woodward test
   */
  KOKKOS_INLINE_FUNCTION
  void initDaiWoodward(Array Q, int i, int j, const DeviceParams &params) {
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    const real_t midbox = 0.5 * (params.xmax + params.xmin);
    const real_t B0 = 1.0 / std::sqrt(4 * M_PI);

    if (x < midbox) {
      Q(j, i, IR) = 1.08;
      Q(j, i, IU) = 1.2;
      Q(j, i, IV) = 0.01;
      Q(j, i, IW) = 0.5;
      Q(j, i, IP) = 0.95;
      Q(j, i, IBY) = B0 * 3.6;
    }
    else {
      Q(j, i, IR) = 1.0;
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IW) = 0.0;
      Q(j, i, IP) = 1.0;
      Q(j, i, IBY) = B0 * 4.0;
    }
    Q(j, i, IBX) = B0 * 4.0;
    Q(j, i, IBZ) = B0 * 2.0;
    Q(j, i, IPSI) = 0.0;
  }

  /**
   * @brief Second Brio-Wu test
   */
  KOKKOS_INLINE_FUNCTION
  void initBrioWu2(Array Q, int i, int j, const DeviceParams &params){
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    const real_t midbox = 0.5 * (params.xmax + params.xmin);

    if (x < midbox) {
      Q(j, i, IR) = 1.0;
      Q(j, i, IP) = 1000.0;
      Q(j, i, IBY) = 1.0;
    }
    else {
      Q(j, i, IR) = 0.125;
      Q(j, i, IP) = 0.1;
      Q(j, i, IBY) = -1.0;
    }
    Q(j, i, IU) = 0.0;
    Q(j, i, IV) = 0.0;
    Q(j, i, IW) = 0.0;
    Q(j, i, IBX) = 0.0;
    Q(j, i, IBZ) = 0.0;
    Q(j, i, IPSI) = 0.0;
  }

  /**
   * @brief Slow Rarefaction Test
   */
  KOKKOS_INLINE_FUNCTION
  void initSlowRarefaction(Array Q, int i, int j, const DeviceParams &params){
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    const real_t midbox = 0.5 * (params.xmax + params.xmin);

    if (x < midbox) {
      Q(j, i, IR) = 1.0;
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = 2.0;
      Q(j, i, IBY) = 0.0;
    }
    else {
      Q(j, i, IR) = 0.2;
      Q(j, i, IU) = 1.186;
      Q(j, i, IV) = 2.967;
      Q(j, i, IP) = 0.1368;
      Q(j, i, IBY) = 1.6405;
    }
    Q(j, i, IW) = 0.0;
    Q(j, i, IBX) = 1.0;
    Q(j, i, IBZ) = 0.0;
    Q(j, i, IPSI) = 0.0;
  }


  /**
   * @brief First Expansion Test
   */
  KOKKOS_INLINE_FUNCTION
  void initExpansion1(Array Q, int i, int j, const DeviceParams &params){
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    const real_t midbox = 0.5 * (params.xmax + params.xmin);

    if (x < midbox) {
      Q(j, i, IU) = -3.1;
    }
    else {
      Q(j, i, IU) = 3.1;
    
    }
    Q(j, i, IR) = 1.0;
    Q(j, i, IV) = 0.0;
    Q(j, i, IW) = 0.0;
    Q(j, i, IP) = 0.45;
    Q(j, i, IBX) = 0.0;
    Q(j, i, IBY) = 0.5;
    Q(j, i, IBZ) = 0.0;
    Q(j, i, IPSI) = 0.0;
  }

  /**
   * @brief Second Expansion Test
   */
  KOKKOS_INLINE_FUNCTION
  void initExpansion2(Array Q, int i, int j, const DeviceParams &params){
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    const real_t midbox = 0.5 * (params.xmax + params.xmin);

    if (x < midbox) {
      Q(j, i, IU) = -3.1;
    }
    else {
      Q(j, i, IU) = 3.1;
    }

    Q(j, i, IR) = 1.0;
    Q(j, i, IV) = 0.0;
    Q(j, i, IW) = 0.0;
    Q(j, i, IP) = 0.45;
    Q(j, i, IBX) = 1.0;
    Q(j, i, IBY) = 0.5;
    Q(j, i, IBZ) = 0.0;
    Q(j, i, IPSI) = 0.0;
  }

  KOKKOS_INLINE_FUNCTION
  void initShuOsher(Array Q, int i, int j, const DeviceParams &params){
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    const real_t x0 = -4.0; // shock interface
    if (x <= x0) {
      Q(j, i, IR) = 3.5;
      Q(j, i, IU) = 5.8846;
      Q(j, i, IV) = 1.1198;
      Q(j, i, IP) = 42.0267;
      Q(j, i, IBY) = 3.6359;
    }
    else {
      Q(j, i, IR) = 1.0 + 0.2 * Kokkos::sin(5.0*x);
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = 1.0;
      Q(j, i, IBY) = 1.0;
    }
    Q(j, i, IW) = 0.0;
    Q(j, i, IBX) = 1.0;
    Q(j, i, IBZ) = 0.0;
    Q(j, i, IPSI) = 0.0;
  }

  KOKKOS_INLINE_FUNCTION
  void initArtificialNonZeroDivB(Array Q, int i, int j, const DeviceParams &params){
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];

    Q(j, i, IR) = 1.0;
    Q(j, i, IU) = 0.0;
    Q(j, i, IV) = 0.0;
    Q(j, i, IW) = 0.0;
    Q(j, i, IP) = 1.0;
    Q(j, i, IBY) = 0.0;
    Q(j, i, IBZ) = 0.0;
    Q(j, i, IPSI) = 0.0;

    if (x <= -0.8)
      Q(j, i, IBX) = 0.0;
    else if (-0.8 < x && x <= -0.6)
      Q(j, i, IBX) = -2.0 * (x + 0.8);
    else if (-0.6 < x && x <= 0.6)
      Q(j, i, IBX) = Kokkos::exp(-0.5 * (x/0.11)*(x/0.11));
    else
      Q(j, i, IBX) = 0.5;
  }
  // 2D MHD Tests
  /**
   * @brief Orszag-Tang vortex
   */
  KOKKOS_INLINE_FUNCTION
  void initOrszagTang(Array Q, int i, int j, const DeviceParams &params){
    const real_t B0 = 1/Kokkos::sqrt(4*M_PI);
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];
  
    Q(j, i, IR) = params.gamma0*params.gamma0*B0*B0;
    Q(j, i, IU) = -sin(2.0*M_PI*y);
    Q(j, i, IV) = sin(2.0*M_PI*x);
    Q(j, i, IW) = 0.0;
    Q(j, i, IP) = params.gamma0*B0*B0;
    Q(j, i, IBX) = -B0*sin(2.0*M_PI*y);
    Q(j, i, IBY) = B0*sin(4.0*M_PI*x);
    Q(j, i, IBZ) = 0.0;
    Q(j, i, IPSI) = 0.0;
  }
  #endif // MHD

  /**
   * @brief Lax-Liu config #13 setup
   */
  KOKKOS_INLINE_FUNCTION
  void initLaxLiu13(Array Q, int i, int j, const DeviceParams &params) {
    real_t xmid = 0.5 * (params.xmin+params.xmax);
    real_t ymid = 0.5 * (params.ymin+params.ymax);

    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];
    real_t rho, u, v, p;

    if (x < xmid) {
      if (y < ymid) {
        rho = 1.0625;
        u   = 0.0;
        v   = 0.8145;
        p   = 0.4;
      }
      else {
        rho = 2.0;
        u   = 0.0;
        v   = 0.3;
        p   = 1.0;
      }
    }
    else {
      if (y < ymid) {
        rho = 0.5313;
        u   = 0.0;
        v   = 0.4276;
        p   = 0.4;
      }
      else {
        rho = 1.0;
        u   = 0.0;
        v   = -0.3;
        p   = 1.0;
      }
    }

    Q(j, i, IR) = rho;
    Q(j, i, IU) = u;
    Q(j, i, IV) = v;
    Q(j, i, IP) = p;
  }

  /**
   * @brief Lax-Liu config #3 setup
   */
  KOKKOS_INLINE_FUNCTION
  void initLaxLiu3(Array Q, int i, int j, const DeviceParams &params) {
    real_t xmid = 0.8;
    real_t ymid = 0.8;

    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    real_t rho, u, v, p;

    if (x < xmid) {
      if (y < ymid) {
        rho = 0.138;
        u   = 1.206;
        v   = 1.206;
        p   = 0.029;
      }
      else {
        rho = 0.5323;
        u   = 1.206;
        v   = 0.0;
        p   = 0.3;
      }
    }
    else {
      if (y < ymid) {
        rho = 0.5323;
        u   = 0.0;
        v   = 1.206;
        p   = 0.3;
      }
      else {
        rho = 1.5;
        u   = 0.0;
        v   = 0.0;
        p   = 1.5;
      }
    }

    Q(j, i, IR) = rho;
    Q(j, i, IU) = u;
    Q(j, i, IV) = v;
    Q(j, i, IP) = p;
  }

  /**
   * @brief Reading a spline from the disk
   **/
   void initProfile(Array Q, const Params &full_params) {
    // Reading input file
    std::string filename = full_params.init_filename;
    std::vector<real_t> y, rho, u, v, p;
    std::ifstream f_in(filename);

    while (f_in.good()) {
      real_t y_, rho_, u_, v_, p_;
      f_in >> y_ >> rho_ >> u_ >> v_ >> p_;
      if (f_in.good()) {
        y.push_back(y_);
        rho.push_back(rho_);
        u.push_back(u_);
        v.push_back(v_);
        p.push_back(p_);
      }
    }
    f_in.close();
    
    // Copying profile on GPU
    size_t N = y.size();
    Kokkos::View<real_t**> profile("profile", N, 5);
    auto profile_host = Kokkos::create_mirror_view(profile);

    std::cout << "Profile read from " << filename << " has " << N << " points" << std::endl;

    for (size_t i=0; i < N; ++i) {
      profile_host(i, 0) = y[i];
      profile_host(i, 1) = rho[i];
      profile_host(i, 2) = u[i];
      profile_host(i, 3) = v[i];
      profile_host(i, 4) = p[i];
    }

    Kokkos::deep_copy(profile, profile_host);

    auto params = full_params.device_params;

    // Initializing domain
    Kokkos::parallel_for("Initialization from profile",
                         full_params.range_dom,
                         KOKKOS_LAMBDA(const int i, const int j) {
                          auto pos = getPos(params, i, j);
                          real_t y = pos[IY];
                    
                          // Finding current cell position in profile.
                          // Could be optimized if dy in the profile is fixed
                          int iy = 0;
                          real_t prof_y = profile(iy, 0);
                          constexpr real_t eps = 1.0e-5;
                          while (prof_y-eps < y && Kokkos::abs(prof_y - y) > eps) {
                            iy++;
                            prof_y = profile(iy, 0);
                          }
                    
                          // Linear interpolation
                          real_t fy = (y - profile(iy-1, 0)) / (profile(iy, 0) - profile(iy-1, 0));
                          for (int ivar=1; ivar < 5; ++ivar)
                            Q(j, i, ivar-1) = profile(iy-1, ivar) * (1.0 - fy) + profile(iy, ivar) * fy;
                        
                          // Todo : Edge case extrapolation
                         });
   }

  /**
   * @brief Gresho-Vortex setup for Low-mach flows
   * 
   * Based on Miczek et al. 2015 "New numerical solver for flows at various Mach numbers"
   */
  KOKKOS_INLINE_FUNCTION
  void initGreshoVortex(Array Q, int i, int j, const DeviceParams &params) {
    Pos pos = getPos(params, i, j);
    const real_t xmid = 0.5*(params.xmin + params.xmax);
    const real_t ymid = 0.5*(params.ymin + params.ymax);
    const real_t xr = pos[IX]-xmid;
    const real_t yr = pos[IY]-ymid;
    const real_t r = sqrt(xr*xr+yr*yr);

    // Pressure is given from density and Mach
    const real_t pressure = params.gresho_density / (params.gamma0 * params.gresho_Mach*params.gresho_Mach);

    Q(j, i, IR) = params.gresho_density;

    real_t u_phi;
    if (r < 0.2) {
      u_phi = 5.0*r;
      Q(j, i, IP) = pressure + 12.5*r*r;
    }
    else if (r < 0.4) {
      u_phi = 2.0 - 5.0*r;
      Q(j, i, IP) = pressure + 12.5*r*r + 4.0*(1.0-5.0*r+log(5.0*r));
    }
    else {
      u_phi = 0.0;
      Q(j, i, IP) = pressure - 2.0 + 4.0*log(2.0);
    }

    const real_t xnr = xr / r;
    const real_t ynr = yr / r;
    Q(j, i, IU) = -ynr * u_phi;
    Q(j, i, IV) =  xnr * u_phi;
    
  }

  /**
   * @brief Implosion setup
   */
   KOKKOS_INLINE_FUNCTION
   void initImplosion(Array Q, int i, int j, const DeviceParams &params) {
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    if (y < x-params.implosion_x0) {
      Q(j, i, IR) = params.implosion_rho_in;
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = params.implosion_p_in;
    }
    else {
      Q(j, i, IR) = params.implosion_rho_out;
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = params.implosion_p_out;
    }
    
   }


}
  

/**
 * @brief Enum describing the type of initialization possible
 */
enum InitType {
  SOD_X,
  SOD_Y,
  BLAST,
  HOT_BUBBLE,
  RAYLEIGH_TAYLOR,
  DIFFUSION,
  H84,
  #ifdef MHD
  MHD_SOD_X,
  MHD_SOD_Y,
  ORSZAG_TANG,
  KELVIN_HELMOLTZ,
  DAI_WOODWARD,
  BRIO_WU2,
  SLOW_RAREFACTION,
  EXPANSION1,
  EXPANSION2,
  SHU_OSHER,
  ARTIFICIAL_NON_ZERO_DIVB,
  BlAST_MHD_STANDARD,
  BlAST_MHD_LOW_BETA,
  ROTATED_SHOCK_TUBE,
  MHD_ROTOR,
  FIELD_LOOP_ADVECTION,
  #endif //MHD
  C91
};

struct InitFunctor {
private:
  Params full_params;
  InitType init_type;
public:
  InitFunctor(Params &full_params)
    : full_params(full_params) {
    std::map<std::string, InitType> init_map {
      {"sod_x", SOD_X},
     // {"sod_x_inverse", SOD_X_INVERSE},
      {"sod_y", SOD_Y},
      {"blast", BLAST},
      {"hot_bubble", HOT_BUBBLE},
      {"rayleigh-taylor", RAYLEIGH_TAYLOR},
      {"diffusion", DIFFUSION},
      {"H84", H84},
      #ifdef MHD
      {"mhd_sod_x", MHD_SOD_X},
      {"mhd_sod_y", MHD_SOD_Y},
      {"orszag-tang", ORSZAG_TANG},
      {"kelvin-helmoltz", KELVIN_HELMOLTZ},
      {"dai-woodward", DAI_WOODWARD},
      {"brio-wu2", BRIO_WU2},
      {"slow-rarefaction", SLOW_RAREFACTION},
      {"expansion1", EXPANSION1},
      {"expansion2", EXPANSION2},
      {"shu-osher", SHU_OSHER},
      {"artifical_non_zero_divB", ARTIFICIAL_NON_ZERO_DIVB},
      {"blast_mhd_standard", BlAST_MHD_STANDARD},
      {"blast_mhd_low_beta", BlAST_MHD_LOW_BETA},
      {"rotated_shock_tube", ROTATED_SHOCK_TUBE},
      {"mhd_rotor", MHD_ROTOR},
      {"field_loop_advection", FIELD_LOOP_ADVECTION},
      #endif //MHD
      {"C91", C91}
    };

    if (init_map.count(full_params.problem) == 0)
      throw std::runtime_error("Error unknown problem " + full_params.problem);

    init_type = init_map[full_params.problem];
  };
  ~InitFunctor() = default;

  void init(Array &Q) {
    auto init_type = this->init_type;
    auto params = full_params.device_params;

    RandomPool random_pool(full_params.seed);

    // Filling active domain ...
    Kokkos::parallel_for( "Initialization", 
                          full_params.range_dom, 
                          KOKKOS_LAMBDA(const int i, const int j) {
                            switch(init_type) {
                              case SOD_X:           initSodX(Q, i, j, params); break;
                              case SOD_Y:           initSodY(Q, i, j, params); break;
                              case BLAST:           initBlast(Q, i, j, params); break;
                              case DIFFUSION:       initDiffusion(Q, i, j, params); break;
                              case HOT_BUBBLE:      initHotBubble(Q, i, j, params); break;
                              case RAYLEIGH_TAYLOR: initRayleighTaylor(Q, i, j, params); break;
                              case H84:             initH84(Q, i, j, params, random_pool); break;
                              
                              #ifdef MHD
                              case MHD_SOD_X:       initMHDSodX(Q, i, j, params); break;
                              case MHD_SOD_Y:       initMHDSodY(Q, i, j, params); break;
                              case ORSZAG_TANG:     initOrszagTang(Q, i, j, params); break;
                              //case KELVIN_HELMOLTZ: initKelvinHelmoltz(Q, i, j, params, random_pool); break;
                              case DAI_WOODWARD:    initDaiWoodward(Q, i, j, params); break;
                              case BRIO_WU2:        initBrioWu2(Q, i, j, params); break;
                              case SLOW_RAREFACTION: initSlowRarefaction(Q, i, j, params); break;
                              case EXPANSION1:      initExpansion1(Q, i, j, params); break;
                              case EXPANSION2:      initExpansion2(Q, i, j, params); break;
                              case SHU_OSHER:       initShuOsher(Q, i, j, params); break;
                              case ARTIFICIAL_NON_ZERO_DIVB: initArtificialNonZeroDivB(Q, i, j, params); break;
                              //case BlAST_MHD_STANDARD: initBlastMHDStandard(Q, i, j, params); break;
                              //case BlAST_MHD_LOW_BETA: initBlastMHDLowBeta(Q, i, j, params); break;
                              //case ROTATED_SHOCK_TUBE: initRotatedShockTube(Q, i, j, params); break;
                              //case MHD_ROTOR:       initMHDRotor(Q, i, j, params); break;
                              //case FIELD_LOOP_ADVECTION: initFieldLoopAdvection(Q, i, j, params); break;
                              #endif //MHD
                              case C91:             initC91(Q, i, j, params, random_pool); break;
                            }
                          });
                          
    // ... and boundaries
    BoundaryManager bc(full_params);
    bc.fillBoundaries(Q, 0.0);
  }

  real_t initGLMch(Array Q, const Params &full_params) const {
  // > Calculate the time step for the GLM wave system, assuming ch=1. 
  // > This is computed only once, then ch is computed frm the current timestep:
  // > dt~1/ch -> dt/dtch1 = 1/ch -> ch=dtch1/dt
    auto params = full_params.device_params;
    real_t lambda_x = 0.0;
    real_t lambda_y = 0.0;
    Kokkos::parallel_reduce("Compute inital GLM Wave Speed",
      full_params.range_dom,
      KOKKOS_LAMBDA(int i, int j, real_t& lambda_x, real_t& lambda_y) {
        State q = getStateFromArray(Q, i, j);
        real_t bx = q[IBX];
        real_t by = q[IBY];
        real_t cs = speedOfSound(q, params);
        real_t va = Kokkos::sqrt((bx*bx + by*by) / (4 * M_PI * q[IR]));
        real_t v_fast = Kokkos::sqrt(0.5 * (va*va + cs*cs + Kokkos::sqrt((va*va + cs*cs)*(va*va + cs*cs) - 4.0*va*va * cs*cs * (bx*bx / (bx*bx + by*by)))));
        real_t lambdaloc_x = Kokkos::abs(q[IU] + fastMagnetoAcousticSpeed(q, params, IX));
        real_t lambdaloc_y = Kokkos::abs(q[IV] + fastMagnetoAcousticSpeed(q, params, IY));
        lambda_x = Kokkos::max(lambda_x, lambdaloc_x);
        lambda_y = Kokkos::max(lambda_y, lambdaloc_y);
      },
      Kokkos::Max<real_t>(lambda_x),
      Kokkos::Max<real_t>(lambda_y)
    );
    return params.CFL * 2.0 / (lambda_x/params.dx + lambda_y/params.dy);
    }
};
}
