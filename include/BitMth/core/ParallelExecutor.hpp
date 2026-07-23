#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include <BitMth/core/types/Worker.hpp>
#include <BitMth/core/config.hpp>

namespace BitMth::core {
  class ParallelExecutor{
    private:

      void* tempTaskData = nullptr;
      void (*tempFunct)(void* , unsigned int, unsigned int) = nullptr;
      void (*tempFunctM)(void* , unsigned int, unsigned int, unsigned int, unsigned int) = nullptr;

      unsigned int numThreads;
      std::unique_ptr<Worker[]> workers;
      std::atomic<bool> running{true};

      void _workerLoop(unsigned int threadId){
        while (running.load(std::memory_order_relaxed)) {

          while (!workers[threadId].hasWork.load(std::memory_order_acquire) && running.load(std::memory_order_relaxed)) {
            std::this_thread::yield();
          }    

          if(!running.load(std::memory_order_relaxed)) break;

          if (this->tempFunct != nullptr) {
            this->tempFunct(tempTaskData, workers[threadId].start,workers[threadId].end);
          }else if(this->tempFunctM != nullptr){
            this->tempFunctM(tempTaskData,
              workers[threadId].rowStart,workers[threadId].rowEnd,  
              workers[threadId].colStart,workers[threadId].colEnd  
            );
          }
          workers[threadId].done.store(true, std::memory_order_release);
          workers[threadId].hasWork.store(false, std::memory_order_release);
        }
      }

      void _join(){
        for (int i = 0; i < numThreads; i++) {
          while (!workers[i].done.load(std::memory_order_acquire)) {
            std::this_thread::yield();
          }
        }
        this->tempTaskData = nullptr;
        this->tempFunct = nullptr;
        this->tempFunctM = nullptr;
      }
    public:

      ParallelExecutor(unsigned int nThreads = std::thread::hardware_concurrency()){
        numThreads = (nThreads == 0) ? config::DEFAULT_NUMBER_THREADS : nThreads;
        workers = std::make_unique<Worker[]>(numThreads);
            
        for (unsigned int i = 0; i < numThreads; i++) {
          workers[i].thr = std::thread(&ParallelExecutor::_workerLoop, this, i);
        }
      }

      ~ParallelExecutor() {
        running.store(false, std::memory_order_relaxed);
        for (unsigned int i = 0; i < numThreads; i++) {
          workers[i].hasWork.store(true, std::memory_order_release); // Despertamos para que vean el apagado
          if (workers[i].thr.joinable()) {
            workers[i].thr.join();
          }
        }
      }


      template<typename T, typename Op>
      void execute(linalg::Matrix<T>& matrix, T scalar, Op funct){
        unsigned int size = matrix.size();
        unsigned int parts = size / numThreads;

        TaskData<T,Op> data{matrix, scalar, funct};
        this->tempTaskData = &data;
        this->tempFunct = [](void* data, unsigned int start, unsigned int end){
          auto* task = static_cast<TaskData<T, Op> *>(data);
          task->funct(task->matrix, task->scalar, start, end);
        };


        for (int i = 0; i < numThreads; i++) {
          workers[i].start = i * parts;
          if(i == numThreads - 1){
            workers[i].end = size;
            workers[i].done.store(false, std::memory_order_relaxed);
            workers[i].hasWork.store(true, std::memory_order_release);
            continue;
          }
          workers[i].end = workers[i].start + parts;
          workers[i].done.store(false, std::memory_order_relaxed);
          workers[i].hasWork.store(true, std::memory_order_release);
        }

        _join();
      }

      template<typename T, typename Op>
      void execute(const linalg::Matrix<T>& matrix, T scalar, Op funct){
        execute(const_cast<linalg::Matrix<T>&>(matrix), scalar, funct);
      }

      template<typename T, typename Op>
      void execute(linalg::Matrix<T>& matrixA, linalg::Matrix<T>& matrixB, Op funct){
        unsigned int totalRows = matrixA.getRows();
        unsigned int colEnd = matrixB.getCols();

        unsigned int activeThreads = numThreads;
        if (totalRows < activeThreads) {
          activeThreads = totalRows;
        }

        if (activeThreads == 0) return;
        unsigned int parts = totalRows / activeThreads;

        TaskDataM<T,Op> data{matrixA, matrixB, funct};
        this->tempTaskData = &data;
        this->tempFunctM = [](void* data, unsigned int startR, unsigned int endR, unsigned int startC, unsigned int endC){
          auto* task = static_cast<TaskDataM<T, Op> *>(data);
          task->funct(task->matrixA, task->matrixB, startR, endR, startC, endC);
        };


        for (int i = 0; i < numThreads; i++) {
          workers[i].rowStart= i * parts;
          workers[i].colStart = 0;
          workers[i].colEnd = colEnd;
          if(i == numThreads - 1){
            workers[i].rowEnd = matrixA.getRows();
          }else{
            workers[i].rowEnd = workers[i].rowStart + parts;
          }
          workers[i].done.store(false, std::memory_order_relaxed);
          workers[i].hasWork.store(true, std::memory_order_release);
        }

        _join();
      }

      template<typename T, typename Op>
      void execute(const linalg::Matrix<T>& matrixA, const linalg::Matrix<T>& matrixB,  Op funct){
        execute(const_cast<linalg::Matrix<T>&>(matrixA), const_cast<linalg::Matrix<T>&>(matrixB), funct);
      }
    };

    inline ParallelExecutor& getParallelExecutor(){
      static ParallelExecutor pe;
      return pe;
    }
}
