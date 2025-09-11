
            # Auto-generated CMake SHCC Compiler Config
            set(CMAKE_SHCC_COMPILER "/home/primrose/Work/SoftHier/gvsoc/third_party/toolchain/install/bin/riscv32-unknown-elf-gcc")
            set(CMAKE_SHCC_COMPILER_WORKS TRUE)
            set(CMAKE_SHCC_FLAGS_INIT "")
            set(SOFTHIER_INSTALL_PATH "/home/primrose/Work/SoftHier/gvsoc/soft_hier/flex_cluster_sdk/runtime/")
            # Compilation rule for SHCC source files to object files
            set(CMAKE_SHCC_COMPILE_OBJECT "<CMAKE_SHCC_COMPILER> <FLAGS> -c <SOURCE> -o <OBJECT>")

            # Linking rule for SHCC object files to executable (using the provided linker script)
            set(CMAKE_SHCC_LINK_EXECUTABLE "<CMAKE_SHCC_COMPILER> <FLAGS> -T /home/primrose/Work/SoftHier/gvsoc/soft_hier/flex_cluster_sdk/runtime//flex_memory.ld <OBJECTS> -o <TARGET>")

            # Additional variables for SRC_DIR and linker script
            set(CMAKE_SHCC_SRC_DIR "")
            set(CMAKE_SHCC_LINKER_SCRIPT "")
        