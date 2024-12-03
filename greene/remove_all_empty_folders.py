import os
dir_of_interest = "./"
empty_dirs = [root for (root,_,_) in os.walk(dir_of_interest, topdown=True) if not os.listdir(root)]
print(empty_dirs)
for d in empty_dirs:
    os.rmdir(d)
