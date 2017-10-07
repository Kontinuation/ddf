.PHONY: run test clean

CC_FLAGS = -g -Wall -Wno-unused-value -std=c++11 -O3 -DDEBUG_BPROP # -DNOTICE_ALLOC_CALL # -fsanitize=address
run:
	g++ -Iinclude src/main.cc src/logging.cc -o bin/run $(CC_FLAGS)

test:
	g++ -Iinclude test/test.cc src/logging.cc -o bin/test $(CC_FLAGS)

clean:
	rm run
