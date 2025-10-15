#include <cstdio>
#include <sstream>
#include <thread>

int main(void) {
  const uint8_t max_threads = std::thread::hardware_concurrency();
  printf("Max threads: %d\n", max_threads);


#pragma omp parallel
  {
    std::stringstream ss;
    ss << std::this_thread::get_id();
    printf("%s, Hello, world.\n", ss.str().c_str());
  }

  return 0;
}
