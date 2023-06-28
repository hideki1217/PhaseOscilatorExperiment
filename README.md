# Phase Oscilator experiment with MCMC

# Requirements

- icpx
- python3
- requirements.txt

# Prepare

- config.yaml

set "gmail" and "gmail_pass" attribute

*config.yaml*
```
gmail: GMAIL
gmail_pass: GMAIL_PASS
```

- phase.param.hpp

set parameter like below

*phase.param.hpp*
``` {c++}
#include <chrono>
#include <fstream>
#include <vector>

// MCMC param
const auto N_sample = 10000;
const auto T_sampling = 200;  // TODO: 決めないと
const static int burn_in = 10000;
const int T_swap = 10;
const auto betas = std::vector<double>({0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6., 10., 14., 20.});
const auto step_size = std::vector<double>(betas.size(), 1.0);
// Phase param
const auto w_left =
    std::vector<double>({0.2682, 0.5776, 0.9995, 1.732, 3.7328});
const double threshold = 0.99;
const int all_steps = 5000;
const double p_eval = 0.2;

void save_param(const char *filename) {
  std::ofstream ofs(filename);
  {
    auto now = std::chrono::system_clock::now();
    std::time_t stamp = std::chrono::system_clock::to_time_t(now);
    ofs << "#" << std::ctime(&stamp) << std::endl;
  }
  ofs << "N_sample: " << N_sample << std::endl;
  ofs << "T_sampling: " << T_sampling << std::endl;

  ofs << "beta: [" << betas[0];
  for (int i = 1; i < betas.size(); i++) ofs << "," << betas[i];
  ofs << "]" << std::endl;

  ofs << "threshold: " << threshold << std::endl;
  ofs << "burn_in: " << burn_in << std::endl;
  ofs << "T_swap: " << T_swap << std::endl;

  ofs << "w0: [" << w_left[0];
  for (int i = 1; i < w_left.size(); i++) ofs << "," << w_left[i];
  ofs << "]" << std::endl;
  
  ofs << "all_steps: " << all_steps << std::endl;
  ofs << "p_eval: " << p_eval << std::endl;
}
```