.PHONY: run test clean

CC_FLAGS = -g -Wall -Wno-unused-value -std=c++11 -O3 -DNDEBUG # -DDEBUG_ACT_DISTRIB # -DDEBUG_BPROP # -DNOTICE_ALLOC_CALL # -fsanitize=address

spiral:
	mkdir -p bin
	g++ -Iinclude sample/spiral.cc src/logging.cc src/dataset.cc -o bin/spiral $(CC_FLAGS)

mnist:
	mkdir -p bin
	g++ -Iinclude sample/mnist.cc src/logging.cc src/dataset.cc -o bin/mnist $(CC_FLAGS)

test:
	mkdir -p bin
	g++ -Iinclude test/test.cc src/logging.cc src/dataset.cc -o bin/test $(CC_FLAGS)

pylib:
	cd python && python setup.py build_ext --inplace

clean:
	rm -rf bin/
