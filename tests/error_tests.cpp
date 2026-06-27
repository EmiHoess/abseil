#include <abseil/error.hpp>

#include <string_view>

int main()
{
  using namespace abseil;

  operation_result<int> value{7};
  if(!value || *value != 7)
    return 1;

  operation_error error{
    operation_code::backend_unavailable,
    operation_stage::dispatch,
    0,
    "backend is unavailable"
  };
  operation_result<int> failed{std::unexpected(error)};
  if(failed)
    return 2;
  if(failed.error().message != "backend is unavailable")
    return 3;
  if(operation_code_name(failed.error().code) != std::string_view{"backend_unavailable"})
    return 4;
  if(operation_stage_name(failed.error().stage) != std::string_view{"dispatch"})
    return 5;

  return 0;
}
