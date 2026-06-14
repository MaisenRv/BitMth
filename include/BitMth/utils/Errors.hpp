#include <sstream>
#include <source_location>

#define DEFAULT_ERROR(errorTag,opt, message) \
    do { \
        std::stringstream ss; \
        ss << "\n\n\033[1;31m["<< #errorTag << "]\033[0m\n" \
        << "  [OPERATION: " << opt << "]\n" \
        << "  MESSAGE: " << message << "\n"; \
        throw std::runtime_error(ss.str()); \
    } while (0)

#define CHECK_ERROR_MATRIX(validExp, opt, message)  \
    do { \
        if ((validExp)) { \
            DEFAULT_ERROR(ERROR_MATRIX, opt, message); \
        } \
    } while (0)

#define CHECK_ERROR_VECTOR(validExp, opt, message, type) \
    do { \
        if ((validExp) && (type == "2d")) { \
            DEFAULT_ERROR(ERROR_VECTOR_2D, opt, message); \
        } \
        if ((validExp) && (type == "3d")) { \
            DEFAULT_ERROR(ERROR_VECTOR_3D, opt, message); \
        } \
    } while (0)


#define CHECK_ERROR_COMPLEX(validExp,opt,message)  \
    do { \
        if ((validExp)) { \
            DEFAULT_ERROR(ERROR_COMPLEX, opt, message); \
        } \
    } while (0)
