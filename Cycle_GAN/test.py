import glob


root = "./data/apple2orange/trainA/*.*"
files_A = sorted(glob.glob(root))

print("git")
print(files_A)