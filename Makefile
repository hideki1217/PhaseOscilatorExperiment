CXX=icpx
CXXFLAGS=-std=c++17 -fast
# CXXFLAGS=-std=c++17 -g

# LDFLAGS=-pg

phase: phase.cpp
	${CXX} ${CXXFLAGS} -qopenmp phase.cpp -o $@.elf

build: phase

clean: 
	rm phase

run: phase
	./phase.sh

asyncrun: phase
	nohup ./phase.sh
