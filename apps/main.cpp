#include "luminis/photon.hpp"
#include <iostream>

int main() {
  luminis::Photon p({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 532.0);
  p.move(10.0);

  std::println("Hello world");
  std::printf("Photon Position: (%.2f, %.2f, %.2f)\n", p.position[0],
              p.position[1], p.position[2]);
}
