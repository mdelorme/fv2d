#include <iostream>
#include <array>

#include "SimInfo.h"
#include "Init.h"
#include "ComputeDt.h"
#include "IOManager.h"
#include "Update.h"

using namespace fv2d;

int main(int argc, char **argv) {
  // Initializing Kokkos
  Kokkos::initialize(argc, argv);
  {
    std::cout << "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░" << std::endl;
    std::cout << "░        ░░   ░░░░░░░░░   ░░░░░░░░░░░░░░░░░   ░" << std::endl;
    std::cout << "▒   ▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒   ▒▒▒   ▒  ▒▒▒▒▒▒▒▒▒   ▒" << std::endl;
    std::cout << "▒   ▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒   ▒▒▒  ▒▒▒▒▒   ▒▒▒▒▒▒   ▒" << std::endl;
    std::cout << "▓       ▓▓▓▓▓▓   ▓▓▓   ▓▓▓▓▓▓▓▓▓   ▓▓▓▓   ▓   ▓" << std::endl;
    std::cout << "▓   ▓▓▓▓▓▓▓▓▓▓▓   ▓   ▓▓▓▓▓▓▓▓   ▓▓▓▓▓  ▓▓▓   ▓" << std::endl;
    std::cout << "▓   ▓▓▓▓▓▓▓▓▓▓▓▓     ▓▓▓▓▓▓▓   ▓▓▓▓▓▓▓  ▓▓▓   ▓" << std::endl;
    std::cout << "█   █████████████   ███████         ███   █   █" << std::endl;
    std::cout << "███████████████████████████████████████████████" << std::endl;

    // Reading parameters from .ini file
    auto params = readInifile(argv[1]);
    std::ofstream out_ini("last.ini");
    params.outputValues(out_ini);
    auto device_params = params.device_params;

    // Allocating main views
    Array U    = Kokkos::View<real_t***>("U", device_params.Nty, device_params.Ntx, Nfields);
    Array Q    = Kokkos::View<real_t***>("Q", device_params.Nty, device_params.Ntx, Nfields);


    real_t GLM_ch1; // GLM wave speed, computed only once
    // Misc vars for iteration
    real_t t = 0.0;
    int save_ite = 0;
    int ite = 0;
    real_t next_save = 0.0;
    
    // Initializing primitive variables
    InitFunctor init(params);
    UpdateFunctor update(params);
    ComputeDtFunctor computeDt(params);
    IOManager ioManager(params);

    if (params.restart_file != "") {
      auto restart_info = ioManager.loadSnapshot(Q);
      if (params.restart_mhd_from_hydro){
        if (!mhd_run)
          throw std::runtime_error("Cannot restart MHD run from hydro if fv2d was not compiled for MHD.");
        else
          std::cout << "Initializing MHD State from saved hydro State" << std::endl;
      init.init(Q);
      GLM_ch1 = init.initGLMch(Q, params);
      }
    t = restart_info.time;
    save_ite = restart_info.iteration;
    std::cout << "Restart at iteration " << ite << " and time " << t << std::endl;
    next_save = t + params.save_freq;
      save_ite++;
    }
    else{
      init.init(Q);
      #ifdef MHD
      GLM_ch1 = init.initGLMch(Q, params);
      #endif // MHD
    }
    primToCons(Q, U, params);

    real_t dt;
    int next_log = 0;

    while (t + device_params.epsilon < params.tend) {
      bool save_needed = (t + device_params.epsilon > next_save);

      dt = computeDt.computeDt(Q, (ite == 0 ? params.save_freq : next_save-t), t, next_log == 0);
      if (next_log == 0)
        next_log = params.log_frequency;
      else
        next_log--;

      if (dt < 1.0e-10) {
        std::cout << "DT too small : Aborting (dt=" << dt << ")" << std::endl;
        break;
      }

      if (save_needed) {
        std::cout << " - Saving at time " << t << std::endl;
        ioManager.saveSolution(Q, save_ite++, t, dt);
        next_save += params.save_freq;

      }

      update.update(Q, U, dt, GLM_ch1);
      consToPrim(U, Q, params);
      checkNegatives(Q, params);

      t += dt;
      ite++;
    }

    std::cout << "Time at end is " << t << std::endl;

    ioManager.saveSolution(Q, save_ite++, t, dt);
  }
  Kokkos::finalize();

  std::cout << std::endl << std::endl;
  std::cout << "    █     ▀██  ▀██         ▀██                              ▄█▄ " << std::endl;
  std::cout << "   ███     ██   ██       ▄▄ ██    ▄▄▄   ▄▄ ▄▄▄     ▄▄▄▄     ███ " << std::endl;
  std::cout << "  █  ██    ██   ██     ▄▀  ▀██  ▄█  ▀█▄  ██  ██  ▄█▄▄▄██    ▀█▀ " << std::endl;
  std::cout << " ▄▀▀▀▀█▄   ██   ██     █▄   ██  ██   ██  ██  ██  ██          █  " << std::endl;
  std::cout << "▄█▄  ▄██▄ ▄██▄ ▄██▄    ▀█▄▄▀██▄  ▀█▄▄█▀ ▄██▄ ██▄  ▀█▄▄▄▀     ▄  " << std::endl;
  std::cout << "                                                            ▀█▀ " << std::endl;

  return 0;
}
