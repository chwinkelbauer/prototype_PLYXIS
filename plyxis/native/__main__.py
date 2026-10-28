import sys

from . import available, build

if len(sys.argv) > 1 and sys.argv[1] == "build":
    sys.exit(0 if build() else 1)
print("native kernels available:", available())
