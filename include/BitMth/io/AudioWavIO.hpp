#pragma once

#include <cstdint>
#include <fstream>
#include <vector>

#include <BitMth/io/interfaces/IWriter.hpp>

namespace BitMth::io {
  #pragma pack(push, 1)
  struct WavData{
    // RIFF
    char id[4]     = {'R','I','F','F'};
    uint32_t size  = 36;
    char format[4] = {'W','A','V','E'};

    // FMT
    char fmt[4]            = {'f','m','t',' '};
    uint32_t fmtSize       = 16; 
    uint16_t audioFormat   = 1;  
    uint16_t numChannels   = 1;  
    uint32_t sampleRate    = 44100;
    uint32_t byteRate      = 0;  
    uint16_t blockAlign    = 0;  
    uint16_t bitsPerSample = 16; 
    
    //DATA
    char     dataId[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize  = 0;  
  };
  #pragma pack(pop)
  
  class AudioWavIO: public interfaces::IWriter<std::vector<int16_t>>{
    public:
      [[nodiscard]] bool write(const std::string &path, const std::vector<int16_t>& data) const override {
        std::ofstream wav(path, std::ios::out | std::ios::binary);

        if(wav.is_open()){
          WavData wData;
          wData.dataSize = data.size() * sizeof(uint16_t);
          wData.size += wData.dataSize;
          wData.blockAlign = wData.numChannels * (wData.bitsPerSample / 8);
          wData.byteRate = wData.sampleRate * wData.blockAlign;
          wav.write(reinterpret_cast<const char*>(&wData), sizeof(wData)); 
          wav.write(reinterpret_cast<const char*>(data.data()), wData.dataSize);
          return true;
        }
        return false;
      };
    
  };
}
