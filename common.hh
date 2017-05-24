#ifndef _COMMON_H_
#define _COMMON_H_

#define DISABLE_COPY_AND_ASSIGN(type)         \
    type(const type &);                       \
    type &operator = (const type &)

#endif /* _COMMON_H_ */
