import cv2
import os
from sklearn.model_selection import train_test_split


DATASET_PATH = 'dataset/datasets/animals/animals/concatenated/'
SAVE_PATH = 'dataset/datasets/animals/'
THRESHOLD1 = 100
THRESHOLD2 = 200
INITIAL_BLUR_WINDOW = (3,3)
SMOOTH_BLUR_WINDOW = (3,3)
RESIZE = (256, 256)
TEST_SIZE = 0.1
split = True


def convert_to_edges(files: list[str]) -> tuple[list, list]:
    labels, sketches = [], []
    for file in files:
        img = cv2.imread(DATASET_PATH + file) 

        img = cv2.resize(img, RESIZE)
        labels.append(img)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, INITIAL_BLUR_WINDOW, 0) 

        edges = cv2.Canny(image=img_gray, threshold1=THRESHOLD1, threshold2=THRESHOLD2)
        edges = cv2.bitwise_not(edges)

        edges = cv2.GaussianBlur(edges, SMOOTH_BLUR_WINDOW, 0)
        sketches.append(edges)
    return labels, sketches

def split_to_train_test(labels: list, sketches: list, path: str) -> None:
    X_train, X_test, y_train, y_test = train_test_split(sketches, labels, test_size=TEST_SIZE)

    for idx, (sketch_train, label_train) in enumerate(zip(X_train, y_train)):
        cv2.imwrite(f'{path}/train/labels/{idx+1}.jpg', label_train)
        cv2.imwrite(f'{path}/train/sketches/{idx+1}.jpg', sketch_train)

    for idx, (sketch_test, label_test) in enumerate(zip(X_test, y_test)):
        cv2.imwrite(f'{path}/test/labels/{idx+1}.jpg', label_test)
        cv2.imwrite(f'{path}/test/sketches/{idx+1}.jpg', sketch_test)

def save_images(labels: list, sketches: list, path: str) -> None:
    for idx, (label, sketch) in enumerate(zip(labels, sketches)):
        cv2.imwrite(f'{path}/labels/{idx+1}.jpg', label)
        cv2.imwrite(f'{path}/sketches/{idx+1}.jpg', sketch)

def create_dirs():
    path = os.path.join(SAVE_PATH, 'processed')
    if not os.path.exists(path):
        os.mkdir(path)

    if split:
        dirs = [os.path.join(path, 'train'),
                os.path.join(path, 'test'),
                os.path.join(path, 'train/labels'),
                os.path.join(path, 'train/sketches'),
                os.path.join(path, 'test/labels'),
                os.path.join(path, 'test/sketches')]
        for dir in dirs:
            if not os.path.exists(dir):
                os.mkdir(dir)
    return path

def main():
    files = os.listdir(DATASET_PATH)

    labels, sketches = convert_to_edges(files)

    path = create_dirs()

    if split:
        split_to_train_test(labels, sketches, path)
    else:
        save_images(labels, sketches, path)


if __name__=='__main__':
    main()
 