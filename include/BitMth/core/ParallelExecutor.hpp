#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include <BitMth/core/types/Worker.hpp>
#include <BitMth/core/config.hpp>

namespace BitMth::core {
  class ParallelExecutor{
    private:
      using FunctPtr = void (*)(void*, unsigned int, unsigned int);
      using FunctMPtr = void (*)(void*, unsigned int, unsigned int, unsigned int, unsigned int);

      std::atomic<void*> tempTaskData{nullptr};
      std::atomic<FunctPtr> tempFunct{nullptr};
      std::atomic<FunctMPtr> tempFunctM{nullptr};

      unsigned int numThreads;
      std::unique_ptr<Worker[]> workers;
      std::atomic<bool> running{true};

      void _workerLoop(unsigned int threadId){
        while (running.load(std::memory_order_relaxed)) {

          while (!workers[threadId].hasWork.load(std::memory_order_acquire)) {
            if (!running.load(std::memory_order_relaxed)) return;
            std::this_thread::yield();
          }    

          auto* f1 = tempFunct.load(std::memory_order_acquire);
          auto* f2 = tempFunctM.load(std::memory_order_acquire);
          void* data = tempTaskData.load(std::memory_order_acquire);

          if (f1 != nullptr) {
            f1(data, workers[threadId].start,workers[threadId].end);
          }else if(f2 != nullptr){
            f2(data,
              workers[threadId].rowStart,workers[threadId].rowEnd,  
              workers[threadId].colStart,workers[threadId].colEnd  
            );
          }
          workers[threadId].hasWork.store(false, std::memory_order_release);
          workers[threadId].done.store(true, std::memory_order_release);
        }
      }

      void _join(unsigned int activeThreads){
        for (unsigned int i = 0; i < activeThreads; i++) {
          while (!workers[i].done.load(std::memory_order_acquire)) {
            std::this_thread::yield();
          }
        }
        tempTaskData.store(nullptr, std::memory_order_relaxed);
        tempFunct.store(nullptr, std::memory_order_relaxed);
        tempFunctM.store(nullptr, std::memory_order_relaxed);
      }

      inline void _setInfoTask(void* data, FunctPtr f1, FunctMPtr f2){
        tempTaskData.store(data, std::memory_order_relaxed);
        tempFunct.store(f1, std::memory_order_release);
        tempFunctM.store(f2, std::memory_order_release);
      }

      inline void _waitForWorkerReady(unsigned int threadId) {
        while (workers[threadId].hasWork.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
      }

      inline void _notifyWorkers(unsigned int activeThreads) {
        for (unsigned int i = 0; i < activeThreads; i++) {
          workers[i].hasWork.store(true, std::memory_order_release);
        }
      }

    public:

      ParallelExecutor(unsigned int nThreads = std::thread::hardware_concurrency()){
        numThreads = (nThreads == 0) ? config::DEFAULT_NUMBER_THREADS : nThreads;
        workers = std::make_unique<Worker[]>(numThreads);
            
        for (unsigned int i = 0; i < numThreads; i++) {
          workers[i].done.store(true, std::memory_order_relaxed);
          workers[i].hasWork.store(false, std::memory_order_relaxed);
          workers[i].thr = std::thread(&ParallelExecutor::_workerLoop, this, i);
        }
      }

      ~ParallelExecutor() {
        running.store(false, std::memory_order_relaxed);
        for (unsigned int i = 0; i < numThreads; i++) {
          workers[i].hasWork.store(true, std::memory_order_release);
          if (workers[i].thr.joinable()) {
            workers[i].thr.join();
          }
        }
      }


      template<typename T, typename Op>
      void execute(linalg::Matrix<T>& matrix, T scalar, Op funct){
        unsigned int size = matrix.size();
        if (size == 0) return;
        unsigned int activeThreads = (size < numThreads) ? size : numThreads;
        unsigned int parts = size / activeThreads;

        TaskData<T,Op> data{matrix, scalar, funct};

        _setInfoTask(
          &data,
          [](void* dataPtr, unsigned int start, unsigned int end) {
            auto* task = static_cast<TaskData<T, Op>*>(dataPtr);
            task->funct(task->matrix, task->scalar, start, end);
          },
          nullptr
        );

        for (unsigned int i = 0; i < activeThreads; i++) {
          _waitForWorkerReady(i);

          workers[i].start = i * parts;
          workers[i].end = (i == activeThreads - 1) ? size : workers[i].start + parts;
          workers[i].done.store(false, std::memory_order_relaxed);
        }

        _notifyWorkers(activeThreads);
        _join(activeThreads);
      }

      template<typename T, typename Op>
      void execute(const linalg::Matrix<T>& matrix, T scalar, Op funct){
        execute(const_cast<linalg::Matrix<T>&>(matrix), scalar, funct);
      }

      template<typename T, typename Op>
      void execute(linalg::Matrix<T>& matrixA, linalg::Matrix<T>& matrixB, Op funct){
        unsigned int totalRows = matrixA.getRows();
        unsigned int colEnd = matrixB.getCols();
        if (totalRows == 0) return;
        unsigned int activeThreads = (totalRows < numThreads) ? totalRows : numThreads;

        unsigned int parts = totalRows / activeThreads;

        TaskDataM<T,Op> data{matrixA, matrixB, funct};

        _setInfoTask(
          &data,
          nullptr,
          [](void* dataPtr, unsigned int startR, unsigned int endR, unsigned int startC, unsigned int endC) {
            auto* task = static_cast<TaskDataM<T, Op>*>(dataPtr);
            task->funct(task->matrixA, task->matrixB, startR, endR, startC, endC);
          }
        );

        for (unsigned int i = 0; i < activeThreads; i++) {
          _waitForWorkerReady(i);

          workers[i].rowStart = i * parts;
          workers[i].rowEnd = (i == activeThreads - 1) ? totalRows : workers[i].rowStart + parts;
          workers[i].colStart = 0;
          workers[i].colEnd = colEnd;
          workers[i].done.store(false, std::memory_order_relaxed);
        }

        _notifyWorkers(activeThreads);
        _join(activeThreads);
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
