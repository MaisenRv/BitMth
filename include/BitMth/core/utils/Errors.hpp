#include <sstream>

#define ERROR_MATRIX(opt,message) ([&] { \
    std::stringstream ss; \
    ss << "\n\n\033[1;31m[ERROR_MATRIX]\033[0m\n" \
       << "  [OPERATION: " << opt << "]\n" \
       << "  MESSAGE: " << message << "\n"; \
    return std::runtime_error(ss.str()); \
}())


#define CHECK_ERROR_MATRIX(validExp, opt, message) \
    do { \
        if ((validExp)) { \
            throw ERROR_MATRIX(opt, message); \
        } \
    } while (0)
