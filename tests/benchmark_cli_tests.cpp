#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const char* message)
{
  if(!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

int invoke(const std::string& executable, const std::string& arguments)
{
  return std::system(("\"" + executable + "\" " + arguments).c_str());
}

}

int main(int argc, char** argv)
{
  require(argc == 2, "benchmark executable argument");
  const std::string executable = argv[1];
  require(invoke(executable, "native_cpu 5 2 1") == 0,
    "valid benchmark invocation");
  require(invoke(executable, "unknown 5 2") != 0,
    "unknown backend rejection");
  require(invoke(executable, "native_cpu five 2") != 0,
    "invalid node count rejection");
  require(invoke(executable, "native_cpu 5 0") != 0,
    "zero repetitions rejection");
  require(invoke(executable, "native_cpu 5 2 one") != 0,
    "invalid warm-up count rejection");
}
