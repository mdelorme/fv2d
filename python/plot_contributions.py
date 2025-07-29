import numpy as numpy
import matplotlib.pyplot as plt
import sys
heating = []
viscous =  []
thermal = []

if '--file' in sys.argv:
    i = sys.argv.index('--file') + 1
    path = sys.argv[i]
else:
    print("[ERROR] Please provide a file to analyse.")
    sys.exit(1)

with open(path, 'r') as f:
    logfile = f.readlines()

def extract_number(text: str) -> float:
    """Extract the number on the line. A typical log line has the form :
    Total heating contribution to energy : -0.106
    """
    try:
        number = line.split(":")[-1].strip()
        number = float(number)
    except ValueError:
        print(f"[ERROR] Could not extract number from line: {line}")
        return 0.0
    return number

for line in logfile:
    if ':' in line:
        if 'heating' in line:
            heating.append(extract_number(line))
        if 'thermal' in line:
            thermal.append(extract_number(line))
        if 'viscous' in line:
            viscous.append(extract_number(line))

plt.figure(figsize=(10, 10))
# plt.title(r"Contributions to Energy for MHD run with $\mathbf{B}=\mathbf{0}$")
plt.title("Contribution to Energy for pure Hydro run (not MHD compiled)")
plt.plot(heating, label='Heating Contribution')
plt.plot(viscous, label='Viscous Contribution')
plt.plot(thermal, label='Thermal Contribution')
plt.legend()
plt.grid()
plt.savefig("contribs_hydro.png")
plt.close()