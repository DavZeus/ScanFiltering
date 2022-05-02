#include <fmt/core.h>

#include "old_logic.h"

int main(int argc, char *argv[]) {

  if (argc < 2) {
    fmt::print("Not enough arguments. File path is needed\n");
    return EXIT_FAILURE;
  }

  old::process(argv[1]);
}
