CXXFLAGS = -O2 -Wall -Wextra -Werror -std=c++11 -pedantic  -pedantic-errors 
INCS = -I include -isystem $(shell root-config --incdir)
LIBS = $(shell root-config --libs)

all: bin/select bin/shuffle_tree

bin/select: obj/select.o | bin
	g++ $(CXXFLAGS) $(INCS) -o $@ $^ $(LIBS)

bin/shuffle_tree: obj/shuffle_tree.o | bin
	g++ $(CXXFLAGS) $(INCS) -o $@ $^ $(LIBS)

obj/%.o: src/%.cxx | obj
	g++ $(CXXFLAGS) $(INCS) -c -o $@ $<

obj:
	mkdir -p obj
bin:
	mkdir -p bin

clean:
	rm -rf obj
	rm -rf bin
	rm -f scripts/*.pyc
