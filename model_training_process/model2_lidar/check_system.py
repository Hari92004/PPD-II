import os
import sys
import subprocess
import torch

print("==========================================================")
print("             SYSTEM HARDWARE & CUDA CONFIGURATION         ")
print("==========================================================")

# 1. PyTorch & CUDA
print(f"Python Executable : {sys.executable}")
print(f"Python Version    : {sys.version.split()[0]}")
print(f"PyTorch Version   : {torch.__version__}")
cuda_avail = torch.cuda.is_available()
print(f"CUDA in PyTorch   : {'ENABLED' if cuda_avail else 'DISABLED'}")

if cuda_avail:
    print(f"CUDA Build Version: {torch.version.cuda}")
    print(f"GPU Device Count  : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        dev_name = torch.cuda.get_device_name(i)
        vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"GPU [{i}]           : {dev_name} ({vram_gb:.2f} GB VRAM)")
else:
    print("PyTorch is running on CPU. Checking if NVIDIA GPU exists...")

# 2. NVIDIA-SMI Check
print("\n--- NVIDIA System Driver (nvidia-smi) ---")
try:
    smi_out = subprocess.check_output("nvidia-smi", shell=True, text=True)
    print(smi_out.strip())
except Exception as e:
    print(f"nvidia-smi could not be executed: {e}")

# 3. CPU & RAM via Windows WMIC / PowerShell
print("\n--- System Processor & Memory ---")
try:
    cpu_out = subprocess.check_output(
        'powershell -Command "Get-CimInstance Win32_Processor | Select-Object -Property Name, NumberOfCores, NumberOfLogicalProcessors | Format-List"',
        shell=True, text=True
    )
    print(cpu_out.strip())
except Exception as e:
    print(f"CPU check error: {e}")

try:
    ram_out = subprocess.check_output(
        'powershell -Command "Get-CimInstance Win32_OperatingSystem | Select-Object @{Name=\'TotalRAM_GB\';Expression={[math]::Round($_.TotalVisibleMemorySize/1MB, 2)}}, @{Name=\'FreeRAM_GB\';Expression={[math]::Round($_.FreePhysicalMemory/1MB, 2)}} | Format-List"',
        shell=True, text=True
    )
    print(ram_out.strip())
except Exception as e:
    print(f"RAM check error: {e}")

print("==========================================================")
