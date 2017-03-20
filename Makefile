SHELL      = /bin/sh
CC         = g++
CUDA_CC    = nvcc

CFLAGS       = -pedantic -Wall -ggdb3 -std=gnu++11 -ffast-math
DEBUGFLAGS   = -O0 -D _DEBUG
RELEASEFLAGS = -O3 -D NDEBUG

TARGET = mult
SOURCES = $(shell echo src/*.cpp)
HEADERS = $(shell echo src/*.hpp)
OBJECTS = $(SOURCES:.cpp=.o)

CUDA_TARGET = cuda_mult
CUDA_SOURCES = $(shell echo cuda_src/*.cu)
CUDA_HEADERS = $(shell echo cuda_src/*.h)

all: $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CC) $(CFLAGS) $(DEBUGFLAGS) -o $(TARGET) $(OBJECTS)

release: $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(RELEASEFLAGS) -o $(TARGET) $(SOURCES)

cuda: 
	$(CUDA_CC) -o $(CUDA_TARGET) $(CUDA_SOURCES)


clean:
	rm -f $(OBJECTS)
	rm -f $(TARGET)
	rm -f $(CUDA_TARGET)