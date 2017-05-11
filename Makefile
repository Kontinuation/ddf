.PHONY: run test clean

CC_FLAGS =  -pg -Wall -Wno-unused-value -std=c++11 -O3
run:
	g++ main.cc logging.cc -o run $(CC_FLAGS)

test:
	g++ test.cc logging.cc -o test $(CC_FLAGS)

clean:
	rm run
