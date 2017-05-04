.PHONY: run clean
run: main.cc
	g++ main.cc -std=c++11 -o run -g -Wall -Wno-unused-value
clean:
	rm run
