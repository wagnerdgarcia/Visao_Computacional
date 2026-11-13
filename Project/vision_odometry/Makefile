.PHONY: all test clean
all:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j2
test: all
	ctest --test-dir build --output-on-failure
clean:
	cmake -E rm -rf build
