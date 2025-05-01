import os
import subprocess

root_paths = [
    "/scratch1/ybudanaz/DaCellerator/AscendCBackend",
    #"/scratch1/ybudanaz/local/ascend-toolkit/latest",
    #"/scratch1/ybudanaz/local/Ascend",
    #"/usr/local/Ascend"
]
symbols_to_check = ["rtLinkedDevBinaryRegister", "__cce_rtLaunch", "_ZTSPFvPvE"]

def deep_search_symbols(symbols, search_paths):
    found_symbols = {}
    for root_path in search_paths:
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.endswith(".so"):
                    lib_path = os.path.join(dirpath, filename)
                    print(f"Try {lib_path}")
                    try:
                        result = subprocess.run(
                            ["nm", "-D", lib_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=5
                        )
                        for symbol in symbols:
                            if symbol in result.stdout:
                                if symbol not in found_symbols:
                                    found_symbols[symbol] = []
                                found_symbols[symbol].append(lib_path)
                    except subprocess.TimeoutExpired:
                        print(f"Timeout checking {lib_path}")
                    except Exception as e:
                        print(f"Error checking {lib_path}: {e}")
    return found_symbols

results = deep_search_symbols(symbols_to_check, root_paths)
for symbol, libs in results.items():
    print(f"Symbol '{symbol}' found in:")
    for lib in libs:
        print(f"  {lib}")