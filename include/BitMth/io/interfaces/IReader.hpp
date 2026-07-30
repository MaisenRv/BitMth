#pragma once

#include <string>

namespace BitMth::io::interfaces {
  template <typename T>
  class IReader{
    public:
      virtual ~IReader() = default;
      [[nodiscard]] virtual T read(const std::string &path) const = 0;
  };
}
