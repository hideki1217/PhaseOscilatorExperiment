CXX=icpx
CXXFLAGS=-std=c++17 -O3
# CXXFLAGS=-std=c++17 -g

# LDFLAGS=-pg

const: const.cpp
	${CXX} ${CXXFLAGS} const.cpp -o $@.elf

mixture: mixture.cpp
	${CXX} ${CXXFLAGS} mixture.cpp -o $@.elf

parallel: parallel.cpp
	${CXX} ${CXXFLAGS} parallel.cpp -o $@.elf

phase: phase.cpp
	${CXX} ${CXXFLAGS} -qopenmp phase.cpp -o $@.elf

build: const mixture parallel phase

clean: 
	rm const mixture parallel phase

run: phase
	nohup ./phase > phase_swap.csv; python3 ./phase.py; python3 ./phase_K.py
