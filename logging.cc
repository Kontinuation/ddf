#include "logging.hh"
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <mutex>


// global mutex for making our synchronous logging methods reentrant
static std::mutex __s_logging_lock;


// terminal controllers/colors
#define TC_RESET     "\e[0m"
#define TC_BOLD      "\e[1m"
#define TC_DIM       "\e[2m"
#define TC_UNDERL    "\e[4m"
#define TC_FORG      "\e[39m"
#define TC_RED       "\e[31m"
#define TC_GREEN     "\e[32m"
#define TC_YELLOW    "\e[33m"
#define TC_BLUE      "\e[34m"
#define TC_CYAN      "\e[36m"


#define BEGIN_LOGGING                           \
	va_list args;								\
	va_start(args, format);						\
    __s_logging_lock.lock()

#define END_LOGGING                             \
	printf(TC_RESET "\n");						\
    __s_logging_lock.unlock();                  \
	va_end (args)

#define LOGGING_INTERFACE(method, heading)                              \
    void logging::method(const char *format, ...) {                     \
        BEGIN_LOGGING;                                                  \
        printf(heading);                                                \
        vprintf(format, args);                                          \
        END_LOGGING;                                                    \
    }


LOGGING_INTERFACE(info,     "[INFO]  " TC_DIM)
LOGGING_INTERFACE(trace,    "[TRACE] " )
LOGGING_INTERFACE(debug,    "[DEBUG] " TC_BOLD)
LOGGING_INTERFACE(warning,  "[WARN]  " TC_YELLOW)
LOGGING_INTERFACE(error,    "[ERROR] " TC_RED)
LOGGING_INTERFACE(critical, "[CRIT]  " TC_RED TC_UNDERL)


void logging::fatal(const char *format, ...)
{
	BEGIN_LOGGING;
	printf("[FATAL] " TC_RED TC_BOLD);
	vprintf(format, args);
	END_LOGGING;
	abort();
}
