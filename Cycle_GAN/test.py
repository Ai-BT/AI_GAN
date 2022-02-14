import glob


root = "./data/apple2orange/trainA/*.*"
files_A = sorted(glob.glob(root))

print(files_A)