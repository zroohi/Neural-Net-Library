# Build with g++
all:
	g++ -std=c++17 "testing.cpp" "src/network.cpp" "src/neuron.cpp" -o nn.exe