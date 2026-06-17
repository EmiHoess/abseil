#include <seil/objective.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

void require(bool condition, const char* message)
{
  if(!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

bool near(double left, double right)
{
  return std::abs(left - right) < 1.0e-12;
}

}

int main()
{
  auto plain = seil::evaluate_loss({}, 2.0);
  require(plain.has_value(), "plain loss accepted");
  require(near(plain->cost, 2.0) && near(plain->weight, 1.0),
    "plain loss values");

  auto huber_edge = seil::evaluate_loss({seil::loss_kind::huber, 2.0}, 2.0);
  auto huber_tail = seil::evaluate_loss({seil::loss_kind::huber, 2.0}, -4.0);
  require(huber_edge.has_value() && near(huber_edge->cost, 2.0),
    "huber threshold cost");
  require(huber_tail.has_value() && near(huber_tail->cost, 6.0) &&
    near(huber_tail->weight, 0.5), "huber tail values");

  auto cauchy_zero = seil::evaluate_loss({seil::loss_kind::cauchy, 3.0}, 0.0);
  require(cauchy_zero.has_value() && near(cauchy_zero->cost, 0.0) &&
    near(cauchy_zero->weight, 1.0), "cauchy zero values");
  auto cauchy = seil::evaluate_loss({seil::loss_kind::cauchy, 2.0}, 2.0);
  require(cauchy.has_value() && near(cauchy->cost, 2.0 * std::log(2.0)) &&
    near(cauchy->weight, 0.5), "cauchy values");

  require(!seil::evaluate_loss({seil::loss_kind::huber, 0.0}, 1.0),
    "zero scale rejected");
  require(!seil::evaluate_loss({}, std::numeric_limits<double>::infinity()),
    "infinite residual rejected");
}
