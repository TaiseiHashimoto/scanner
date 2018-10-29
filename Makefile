CXX = g++
WALL = 
CXXFLAGS = `pkg-config opencv --cflags` -g
LDLIBS= `pkg-config opencv --libs`

PROGNAME = detect
BINDIR = bin
TARGET = $(BINDIR)/$(PROGNAME)
SOURCES = $(wildcard *.cpp)
OBJECTS = $(addprefix $(BINDIR)/, $(SOURCES:.cpp=.o))
HEADER = detect.hpp

.PHONY: clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(WALL) -o $@ $^ $(LDLIBS)

$(BINDIR)/%.o: %.cpp $(HEADER)
	$(CXX) $(WALL) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -f $(OBJECTS) $(TARGET)
