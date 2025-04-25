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

    // Allocating main views
    Array U    = Kokkos::View<real_t***>("U",    params.Nty, params.Ntx, Nfields);
    Array Q    = Kokkos::View<real_t***>("Q",    params.Nty, params.Ntx, Nfields);


    // Misc vars for iteration
    real_t t = 0.0;
    int ite = 0;
    real_t next_save = 0.0;
    
    // Initializing primitive variables
    InitFunctor init(params);
    UpdateFunctor update(params);
    ComputeDtFunctor computeDt(params);
    IOManager ioManager(params);

    if (params.restart_file != "") {
      auto restart_info = ioManager.loadSnapshot(Q);
      t = restart_info.time;
      ite = restart_info.iteration;
      std::cout << "Restart at iteration " << ite << " and time " << t << std::endl;
      next_save = t + params.save_freq;
      ite++;
    }
    else
      init.init(Q);
    primToCons(Q, U, params);

    real_t dt;
    int next_log = 0;

    while (t + params.epsilon < params.tend) {
      bool save_needed = (t + params.epsilon > next_save);

      dt = computeDt.computeDt(Q, (ite == 0 ? params.save_freq : next_save-t), t, next_log == 0);
      if (next_log == 0)
        next_log = params.log_frequency;
      else
        next_log--;

      if (save_needed) {
        std::cout << " - Saving at time " << t << std::endl;
        ioManager.saveSolution(Q, ite++, t, dt);
        next_save += params.save_freq;
      }

      update.update(Q, U, dt);
      consToPrim(U, Q, params);
      checkNegatives(Q, params);

      t += dt;
    }

    std::cout << "Time at end is " << t << std::endl;

    ioManager.saveSolution(Q, ite++, t, dt);
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