#ifndef __MLPACK_CORE_OPTIMIZERS_ADADELTA_ADADELTA_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_ADADELTA_ADADELTA_IMPL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
class AdaDelta
{
 public:
  AdaDelta(DecomposableFunctionType& function,
          const double rho = 0.95,
          const double eps = 1e-6) :
          const size_t maxIterations = 100000,
          const double tolerance = 1e-5,
          const bool shuffle = true);
          
  double Optimize(arma::mat& iterate);
  
  const DecomposableFunctionType& Function() const { return function; }
  
  DecomposableFunctionType& Function() { return function; }
  
  double Rho() const { return rho; }

  double& Rho() { return rho; }
  
  //! Get the value used to initialise the mean squared gradient parameter.
  double Epsilon() const { return eps; }
  //! Modify the value used to initialise the mean squared gradient parameter.
  double& Epsilon() { return eps; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return shuffle; }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return shuffle; }
  
 private:
  //! The instantiated function.
  DecomposableFunctionType& function;

  double rho;

  //! The value used to initialise the mean squared gradient parameter.
  double eps;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  bool shuffle;
};
  
} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "ada_delta_impl.hpp"

#endif  
