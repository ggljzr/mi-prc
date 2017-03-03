SHELL      = /bin/sh
CC         = g++

CFLAGS       = -pedantic -Wall -ggdb3
DEBUGFLAGS   = -O0 -D _DEBUG
RELEASEFLAGS = -O2 -D NDEBUG

TARGET = mult
SOURCES = $(shell echo src/*.cpp)
HEADERS = $(shell echo src/*.hpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CC) $(CFLAGS) $(DEBUGFLAGS) -o $(TARGET) $(OBJECTS)

release: $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(RELEASEFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(OBJECTS)
	rm -f $(TARGET)
	rm -f gmon.put
