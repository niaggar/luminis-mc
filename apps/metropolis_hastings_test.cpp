#include <luminis/sample/meanfreepath.hpp>

int main() {
  using namespace luminis::sample;

  ExpDistribution exp_dist(1.0); // mean free path = 1.0

  metropolis_hastings mh(&exp_dist);

  return 0;
}
