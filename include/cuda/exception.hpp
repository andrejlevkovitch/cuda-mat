// exception.hpp

#pragma once

#include <string>


namespace cuda {
/**\brief cuda exception
 */
class exception : public std::exception {
public:
  exception(std::string what)
      : what_{std::move(what)} {
  }

  const char *what() const noexcept override {
    return what_.c_str();
  }

private:
  std::string what_;
};
} // namespace cuda
