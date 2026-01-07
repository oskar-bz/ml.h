import os

with open("lib/ml.h", "r") as f:
    header = f.read()

with open("lib/ml.c", "r") as f:
    source = f.read()

with open("ml.h", "w") as f:
    f.write(header)
    f.write("#ifdef ML_IMPLEMENTATION\n")
    f.write(source)
    f.write("#endif\n")
