.PHONY: all debug release clean check

# Default target
all: debug

# Build Debug preset and symlink compile_commands.json
debug:
	@echo "Configuring Debug preset..."
	cmake --preset debug -DGPT2LAB_ENABLE_CUDA=ON
	@echo "Building Debug..."
	cmake --build build/debug -j$$(nproc)
	@echo "Symlinking compile_commands.json to root..."
	@ln -sf build/debug/compile_commands.json compile_commands.json

# Build Release preset and symlink compile_commands.json
release:
	@echo "Configuring Release preset..."
	cmake --preset release -DGPT2LAB_ENABLE_CUDA=ON
	@echo "Building Release..."
	cmake --build build/release -j$$(nproc)
	@echo "Symlinking compile_commands.json to root..."
	@ln -sf build/release/compile_commands.json compile_commands.json

# Run tests
check: debug
	@echo "Running tests..."
	./build/debug/test_runner

# Clean build directory
clean:
	@echo "Cleaning build directory..."
	rm -rf build/
	rm -f compile_commands.json
