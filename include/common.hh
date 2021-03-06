#ifndef _COMMON_H_
#define _COMMON_H_

#include <exception>

#define DISABLE_COPY_AND_ASSIGN(type)         \
    type(const type &);                       \
    type &operator = (const type &)

namespace ddf {

enum mode {
    TRAINING,
    PREDICT,
};

class exception : public std::exception {
public:
    explicit exception(const std::string& message):
        _msg(message) {
    }

    virtual ~exception(void) = default;

    virtual const char *what() const noexcept {
        return _msg.c_str();
    }

protected:
    std::string _msg;
};

// type to id
template <typename numeric_type>
struct numeric_type_id {
};

template <> struct numeric_type_id<float> {
    static const unsigned int value = 1;
};

template <> struct numeric_type_id<double> {
    static const unsigned int value = 2;
};

} // end namespace ddf

#endif /* _COMMON_H_ */
