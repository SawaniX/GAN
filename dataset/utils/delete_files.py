import os

to_delete = [8, 9, 10, 26, 60, 84, 89, 99, 103, 106, 122, 124, 131, 145, 157, 177, 179, 181, 187, 188, 198, 203, 206, 211, 225]
PATH = 'dataset/datasets/animals/processed/test/'

for name in to_delete:
    os.remove(f"{PATH}labels/{name}.jpg")
    os.remove(f"{PATH}sketches/{name}.jpg")