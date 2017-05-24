.PHONY: run test clean

CC_FLAGS = -g -Wall -Wno-unused-value -std=c++11 -O3 -DNDEBUG # -DNOTICE_ALLOC_CALL # -fsanitize=address
run:
	g++ main.cc logging.cc -o run $(CC_FLAGS)

test:
	g++ test.cc logging.cc -o test $(CC_FLAGS)

clean:
	rm run
