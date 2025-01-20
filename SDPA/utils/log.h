#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>
#include <pthread.h> // 可选，用于线程安全

// 日志级别定义
typedef enum {
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR,
} LogLevel;

// 当前日志级别（低于此级别的日志不会输出）
static LogLevel CURRENT_LOG_LEVEL = LOG_LEVEL_DEBUG;

// 日志文件路径
#define LOG_FILE_PATH "/var/log/sdpa.log"

// 可选：多线程环境下启用日志文件写入锁
static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

// 日志级别名称
static const char *LOG_LEVEL_NAMES[] = {
    "DEBUG",
    "INFO",
    "WARN",
    "ERROR"
};

// LOG_UTIL 宏
#define LOG_UTIL(level, format, ...) do {                                \
    if (level >= CURRENT_LOG_LEVEL) {                                   \
        LogUtil(level, __FILE__, __LINE__, __FUNCTION__, format, ##__VA_ARGS__); \
    }                                                                   \
} while (0)

// 日志工具函数
void LogUtil(LogLevel level, const char *file, int line, const char *func, const char *format, ...) {
    FILE *log_file = fopen(LOG_FILE_PATH, "a");
    if (!log_file) {
        perror("Failed to open log file");
        return;
    }

    // 获取当前时间
    time_t t = time(NULL);
    struct tm *tm_info = localtime(&t);
    char time_buffer[26];
    strftime(time_buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);

    // 格式化日志内容
    va_list args;
    va_start(args, format);

    // 可选：线程安全
    pthread_mutex_lock(&log_mutex);

    fprintf(log_file, "[%s] [%s] %s:%d:%s(): ", time_buffer, LOG_LEVEL_NAMES[level], file, line, func);
    vfprintf(log_file, format, args);
    fprintf(log_file, "\n");

    pthread_mutex_unlock(&log_mutex);

    va_end(args);

    fclose(log_file);
}
