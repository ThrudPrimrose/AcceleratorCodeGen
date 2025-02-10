// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
/**
 * Stub library that can load other libraries for use in as DaCe programs
 **/


#ifdef _WIN32
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

// Loads a library and returns a handle to it, or NULL if there was an error
// NOTE: On Windows, path must be given as a Unicode string (UTF-16, or
//       ctypes.c_wchar_p)
extern "C" void *load_library(const char *filename) {
    if (!filename)
        return nullptr;

    void *hLibrary = nullptr;

#ifdef _WIN32
    hLibrary = (void *)LoadLibraryW((const wchar_t*)filename);
#else
    hLibrary = dlopen(filename, RTLD_LOCAL | RTLD_NOW);
#endif

    return hLibrary;
}

// Returns 1 if the library is already loaded, 0 if not, or -1 on error
extern "C" int is_library_loaded(const char *filename) {
    if (!filename)
        return -1;

    void *hLibrary = nullptr;

#ifdef _WIN32
    hLibrary = (void *)GetModuleHandleW((const wchar_t*)filename);
#else
    hLibrary = dlopen(filename, RTLD_LOCAL | RTLD_NOW | RTLD_NOLOAD);
#endif

    if (hLibrary)
        return 1;
    return 0;
}

// Loads a library function and returns a pointer, or NULL if it was not found
extern "C" void *get_symbol(void *hLibrary, const char *symbol) {
    if (!hLibrary || !symbol)
        return nullptr;

    void *sym = nullptr;

#ifdef _WIN32
    sym = GetProcAddress((HMODULE)hLibrary, symbol);
#else
    sym = dlsym(hLibrary, symbol);
#endif

    return sym;
}

// Loads a library and returns a handle to it, or NULL if there was an error
// NOTE: On Windows, path must be given as a Unicode string (UTF-16, or
//       ctypes.c_wchar_p)
extern "C" void unload_library(void *hLibrary) {
    if (!hLibrary)
        return;

#ifdef _WIN32
    FreeLibrary((HMODULE)hLibrary);
#else
    dlclose(hLibrary);
#endif
}
