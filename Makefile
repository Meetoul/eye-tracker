CXX=g++
CXXFLAGS=-std=c++14 -march=native -Ofast

OUT=tracker
SRC=main.cpp

LIBS=opencv4

LIB_PARAMS=$(shell pkg-config --cflags --libs $(LIBS)) -ldlib -llapack -lcblas -lpthread

all: $(OUT)

clean:
	rm $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(LIB_PARAMS) -o $@ $^
