/***
 * @file io_deleter.cc
 * @author Ryan Curtin
 *
 * Extremely simple class whose only job is to delete the existing CLI object at
 * the end of execution.  This is meant to allow the user to avoid typing
 * 'CLI::Destroy()' at the end of their program.  The file also defines a static
 * CLIDeleter class, which will be initialized at the beginning of the program
 * and deleted at the end.  The destructor destroys the CLI singleton.
 */
#include "io_deleter.h"
#include "io.h"
#include "../io/log.h"

using namespace mlpack;
using namespace mlpack::io;

/***
 * Empty constructor that does nothing.
 */
CLIDeleter::CLIDeleter() {
  /* nothing to do */
}

/***
 * This destructor deletes the CLI singleton.
 */
CLIDeleter::~CLIDeleter() {
  // Delete the singleton!
  CLI::Destroy();
}
