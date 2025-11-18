#ifndef OMP_UTILS_HPP
#define OMP_UTILS_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <iostream>
#include <string>

// ============================================================================
// MACROS DE TIMING SIMPLES
// ============================================================================

#define TIMER_START(var_name) \
    auto _start_##var_name = std::chrono::high_resolution_clock::now()

#define TIMER_END(var_name) \
    auto _end_##var_name = std::chrono::high_resolution_clock::now(); \
    auto _duration_##var_name = std::chrono::duration_cast<std::chrono::milliseconds>(\
        _end_##var_name - _start_##var_name)

#define TIMER_PRINT(var_name, message) \
    std::cout << "[TIMER] " << message << ": " << _duration_##var_name.count() << "ms" << std::endl

#define TIMER_GET_MS(var_name) \
    std::chrono::duration_cast<std::chrono::milliseconds>(\
        std::chrono::high_resolution_clock::now() - _start_##var_name).count()

// ============================================================================
// MACROS AVEC CONTEXTE AUTOMATIQUE
// ============================================================================

#define TIMED_SCOPE(var_name, message) \
    TimedScope _timed_scope_##var_name(message)

#define TIMED_FUNCTION() \
    TimedScope _timed_scope_function(__FUNCTION__)

// ============================================================================
// CLASSES UTILITAIRES
// ============================================================================

class TimedScope {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::string message_;
    
public:
    TimedScope(const std::string& msg) : message_(msg) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    ~TimedScope() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        std::cout << "[TIMED] " << message_ << ": " << duration.count() << "ms" << std::endl;
    }
    
    long long elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_).count();
    }
};

// ============================================================================
// FONCTIONS DE TIMING MANUELLES
// ============================================================================

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    
public:
    Timer() { reset(); }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    long long elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_).count();
    }
    
    long long elapsed_us() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count();
    }
    
    double elapsed_sec() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(now - start_).count();
    }
};

// ============================================================================
// UTILITAIRES OPENMP
// ============================================================================

namespace omp_utils {

#ifdef _OPENMP
    inline int get_max_threads() { return omp_get_max_threads(); }
    inline int get_thread_num() { return omp_get_thread_num(); }
    inline int get_num_threads() { return omp_get_num_threads(); }
    inline void set_num_threads(int n) { omp_set_num_threads(n); }
    inline bool in_parallel() { return omp_in_parallel(); }
#else
    inline int get_max_threads() { return 1; }
    inline int get_thread_num() { return 0; }
    inline int get_num_threads() { return 1; }
    inline void set_num_threads(int n) { (void)n; } // No-op
    inline bool in_parallel() { return false; }
#endif

    inline void print_omp_config() {
#ifdef _OPENMP
        std::cout << "[OMP] Config: " << get_max_threads() << " threads disponibles" << std::endl;
#else
        std::cout << "[OMP] Config: Mode sequentiel (OpenMP desactive)" << std::endl;
#endif
    }

    // Benchmarks automatiques séquentiel vs parallèle
    template<typename Func>
    static double benchmark(const std::string& name, Func&& func, int iterations = 1) {
        Timer timer;
        
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        
        double total_time = timer.elapsed_sec();
        double avg_time = total_time / iterations;
        
        std::cout << "[BENCH] " << name << ": " << avg_time * 1000 << " ms/iter";
        if (iterations > 1) {
            std::cout << " (" << iterations << " iterations)";
        }
        std::cout << std::endl;
        
        return avg_time;
    }
}

#endif // OMP_UTILS_HPP
