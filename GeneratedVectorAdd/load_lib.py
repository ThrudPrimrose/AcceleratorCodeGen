import ctypes
import os

# Path to the libraries (adjust these paths to match your specific library locations)
stub_library_path = './build/libstub_library.so'  # Use .dll on Windows
runner_lib_path = './build/librunner_lib_npu.so'  # Use .dll on Windows
kernel_lib_path = './build/libascendc_kernels.so'

ascend_home = os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend')
runtime_lib_path = os.path.join(ascend_home, 'runtime', 'lib64', 'libruntime.so')
acl_lib_path = os.path.join(ascend_home, 'runtime', 'lib64', 'libascendcl.so')
try:
    runtime_lib = ctypes.CDLL(runtime_lib_path, mode=ctypes.RTLD_GLOBAL)
    print(f"Loaded runtime library from {runtime_lib_path}")
    acl_lib = ctypes.CDLL(acl_lib_path, mode=ctypes.RTLD_GLOBAL)
    print(f"Loaded kernel library from {acl_lib_path}")
    ascendc_lib = ctypes.CDLL(kernel_lib_path, mode=ctypes.RTLD_GLOBAL)
    print(f"Loaded kernel library from {kernel_lib_path}")
except Exception as e:
    print(f"Failed to load runtime library: {e}")
    exit(1)

# Load the stub library
stub_lib = ctypes.CDLL(stub_library_path)

# Function prototypes for stub library functions
stub_lib.load_library.argtypes = [ctypes.c_char_p]
stub_lib.load_library.restype = ctypes.c_void_p

stub_lib.get_symbol.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
stub_lib.get_symbol.restype = ctypes.c_void_p

try:
    # Check if library is already loaded
    is_loaded = stub_lib.is_library_loaded(runner_lib_path.encode('utf-8'))
    print(f"Is library already loaded? {is_loaded}")

    # Try loading the library
    runner_lib_handle = ctypes.CDLL(runner_lib_path)
    print(f"Successfully loaded {runner_lib_path}")
except Exception as e:
    print(f"Library loading error: {e}")

    # On Linux, you can use dlerror to get more detailed error information
    import ctypes.util
    dlerror = ctypes.CDLL(ctypes.util.find_library('dl')).dlerror
    dlerror.restype = ctypes.c_char_p

    error_msg = dlerror()
    if error_msg:
        print(f"Detailed error: {error_msg.decode('utf-8')}")

# Get the main function from the runner library
try:
    # Use the library's _handle attribute directly
    main_func_ptr = stub_lib.get_symbol(runner_lib_handle._handle, b'main')

    if main_func_ptr:
        # Convert function pointer to callable
        main_func_type = ctypes.CFUNCTYPE(ctypes.c_int)
        main_func = main_func_type(main_func_ptr)

        # Call the main function
        result = main_func()
        print(f"Main function returned: {result}")
    else:
        print("Failed to get main function pointer")

except Exception as e:
    print(f"Error getting main symbol: {e}")

    # Additional debugging
    import traceback
    traceback.print_exc()

if not main_func:
    print("Failed to find main function")
    exit(1)

# Convert the function pointer to a callable Python function
# Adjust the argument types and return type as needed
main_func_type = ctypes.CFUNCTYPE(ctypes.c_int)
main_func_callable = main_func_type(main_func)

# Call the main function
result = main_func_callable()
print(f"Main function returned: {result}")