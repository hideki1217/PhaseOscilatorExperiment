CXX=icpx
CXXFLAGS=-std=c++17 -fast
# CXXFLAGS=-std=c++17 -g

# LDFLAGS=-pg

const: const.cpp
	${CXX} ${CXXFLAGS} const.cpp -o $@.elf

mixture: mixture.cpp
	${CXX} ${CXXFLAGS} mixture.cpp -o $@.elf

parallel: parallel.cpp
	${CXX} ${CXXFLAGS} parallel.cpp -o $@.elf