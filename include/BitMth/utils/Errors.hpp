#include <sstream>
#include <source_location>

#define CHECK_ERROR_MATRIX(validExp, opt, message)  \
    do { \
        if ((validExp)) { \
            std::stringstream ss; \
            ss << "\n\n\033[1;31m[ERROR_MATRIX]\033[0m\n" \
            << "  [OPERATION: " << opt << "]\n" \
            << "  MESSAGE: " << message << "\n"; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (0)


#define ERROR_VECTOR_2D(opt,message) ([&] { \
    std::stringstream ss; \
    ss << "\n\n\033[1;31m[ERROR_VECTOR_2D]\033[0m\n" \
        << "    [OPERATION: " << opt << "]\n" \
        << "    MESSAGE: " << message << "\n"; \
    return std::runtime_error(ss.str()); \
}())

#define CHECK_ERROR_VECTOR(validExp, opt, message, type) \
    do { \
        if ((validExp) && (type == "2d")) { \
            throw ERROR_VECTOR_2D(opt, message); \
        } \
        if ((validExp) && (type == "3d")) { \
            throw ERROR_VECTOR_2D(opt, message); \
        } \
    } while (0)


#define CHECK_ERROR_COMPLEX(validExp,opt,message)  \
    do { \
        if ((validExp)) { \
            std::stringstream ss; \
            ss << "\n\n\033[1;31m[ERROR_COMPLEX]\033[0m\n" \
                << "    [OPERATION: " << opt << "]\n" \
                << "    MESSAGE: " << message << "\n"; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (0)
