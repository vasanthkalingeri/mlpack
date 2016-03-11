#ifndef __MLPACK_CORE_OPTIMIZERS_ADAM_ADAM_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_ADAM_ADAM_IMPL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
class Adam
{
 public:

  Adam(DecomposableFunctionType& function,
      const double stepSize = 0.001,
      const double beta1 = 0.9,
      const double beta2 = 0.999,
      const double eps = 1e-8,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5,
      const bool shuffle = true);
    
  double Optimize(arma::mat& iterate);
    
  //! Get the instantiated function to be optimized.
  const DecomposableFunctionType& Function() const { return function; }
  //! Modify the instantiated function.
  DecomposableFunctionType& Function() { return function; }
  
  //! Get the first smoothing parameter.
  double Beta1() const { return beta1; }
  //! Modify the first smoothing parameter.
  double& Beta1() { return beta1; }
  
  //! Get the second smoothing parameter.
  double Beta2() const { return beta2; }
  //! Modify the second smoothing parameter.
  double& Beta2() { return beta2; }
  
  //! Get the step size.
  double StepSize() const { return stepSize; }
  //! Modify the step size.
  double& StepSize() { return stepSize; }
  
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

  //! The step size for each example.
  double stepSize;

  //! The smoothing parameter.
  double beta1;
  
  //! The smoothing parameter
  double beta2;

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
#include "adam_impl.hpp"

#endif
