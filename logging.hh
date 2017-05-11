#ifndef _LOGGING_H_
#define _LOGGING_H_


// 
// Collection of static methods for dumping logs in various levels
// TODO: add set_log_level function to inhibit low priority logs
// 
class logging
{
public:
	static void info(const char *format, ...);
	static void trace(const char *format, ...);
	static void debug(const char *format, ...);
	static void warning(const char *format, ...);
	static void error(const char *format, ...);
	static void critical(const char *format, ...);
	static void fatal(const char *format, ...);
};


#endif /* _LOGGING_H_ */
