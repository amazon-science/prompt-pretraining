import sys

cfg_path = sys.argv[1]
print(cfg_path.split(".")[0].replace("/", "_"))
