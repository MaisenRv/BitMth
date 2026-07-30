#pragma once

#include <cstdint>
#include <string>
#include <fstream>
#include <vector>

#include <BitMth/io/interfaces/IReader.hpp>
#include <BitMth/linalg/Matrix.hpp>

namespace BitMth::io {
  #pragma pack(push,1)
  struct NpyHeader{
    char magicString[6] = {'\x93','N','U','M','P','Y'};
    uint8_t majorVersion = 1;
    uint8_t minorVersion = 0;
    uint16_t headerLength = 0;
  };
  #pragma pack(pop)

  struct NpyData{
    NpyHeader header;
    std::string dictHeader;
    std::vector<uint8_t> data;
  };

  template<typename T>
  class DataNpyIO: public interfaces::IReader<linalg::Matrix<T>> {
    public:
      [[nodiscard]] linalg::Matrix<T> read(const std::string &path) const override{
         std::ifstream file(path, std::ios::binary);
          if (!file.is_open()) {
            throw std::runtime_error("No se pudo abrir el archivo .npy en la ruta especificada.");
          }

          NpyHeader fixedHeader;
          file.read(reinterpret_cast<char*>(&fixedHeader), sizeof(NpyHeader));
      
          if (fixedHeader.magicString[0] != '\x93' || fixedHeader.magicString[1] != 'N' || 
              fixedHeader.magicString[2] != 'U'   || fixedHeader.magicString[3] != 'M' || 
              fixedHeader.magicString[4] != 'P'   || fixedHeader.magicString[5] != 'Y') {
            throw std::runtime_error("El archivo no tiene un formato .npy valido (Magic String incorrecto).");
          }

          size_t dictLength = fixedHeader.headerLength;
          if (fixedHeader.majorVersion == 2) {
            file.seekg(8, std::ios::beg);
            uint32_t extendedLength = 0;
            file.read(reinterpret_cast<char*>(&extendedLength), sizeof(uint32_t));
            dictLength = extendedLength;
          }

          std::string dictHeader(dictLength, '\0');
          file.read(&dictHeader[0], dictLength);

          size_t shapePos = dictHeader.find("'shape'");
          if (shapePos == std::string::npos) {
            throw std::runtime_error("No se encontro la clave 'shape' en el encabezado del archivo.");
          }

          size_t openParenthesis = dictHeader.find('(', shapePos);
          size_t closeParenthesis = dictHeader.find(')', openParenthesis);
          if (openParenthesis == std::string::npos || closeParenthesis == std::string::npos) {
            throw std::runtime_error("Formato de tupla 'shape' invalido.");
          }

          std::string shapeContent = dictHeader.substr(openParenthesis + 1, closeParenthesis - openParenthesis - 1);
          std::stringstream ss(shapeContent);
      
          size_t rows = 0;
          size_t cols = 1;
          char comma;

          ss >> rows;
          if (ss >> comma) { 
            if (!(ss >> cols)) {
              cols = 1; 
            }
          }

          linalg::Matrix<T> matrix(rows, cols, nullptr, false);
          size_t totalBytes = matrix.size() * sizeof(T);

          file.read(reinterpret_cast<char*>(matrix.getValues()),totalBytes );

          if (!file && file.gcount() < static_cast<std::streamsize>(totalBytes)) {
            throw std::runtime_error("El archivo termino inesperadamente antes de leer todos los datos del array.");
          }
          return matrix;
      }
  };
}
