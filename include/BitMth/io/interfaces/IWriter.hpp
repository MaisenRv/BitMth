#pragma once

#include <string>

namespace BitMth::io::interfaces {
  template <typename T>
  class IWriter{
    public:
      virtual ~IWriter() = default;
      [[nodiscard]] virtual bool write(const std::string &path, const T& data) const = 0;
  };
}
