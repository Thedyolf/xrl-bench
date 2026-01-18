import sys, site, pkgutil
print("Interpreter:", sys.executable)
print("Version:", sys.version)
print("Site-packages:", site.getsitepackages() if hasattr(site,'getsitepackages') else 'n/a')
print("User-site:", site.getusersitepackages())
print("First 10 sys.path entries:")
for p in sys.path[:10]: print("  ", p)
print("\nIs requests installed?", "requests" in {m.name for m in pkgutil.iter_modules()})
