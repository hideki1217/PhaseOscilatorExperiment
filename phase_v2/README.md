# Opy: Oscilator Network Library for Python

## supported

- [ ] Fast phase order calculator with convergence check.
- [ ] etc...

# Setup for User

1. Run the scripts below at "."
```
./setup.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build 
```
2. Then shared library named "opy.*.so" is created in the "./py" directory.
3. Use it !! At vscode, any intelisense is not working about Opy, so reference "/src/opy.cpp"


