import numpy as np
import struct
from pathlib import Path
from typing import Tuple, Union
from numpy.typing import NDArray

UInt8Array = NDArray[np.uint8]
PathLike = Union[str, Path]


class MnistDataloader:
    def __init__(
        self,
        training_images_filepath: PathLike,
        training_labels_filepath: PathLike,
        test_images_filepath: PathLike,
        test_labels_filepath: PathLike,
    ) -> None:
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(
        self,
        images_filepath: PathLike,
        labels_filepath: PathLike,
    ) -> Tuple[UInt8Array, UInt8Array]:

        # ---- Load labels ----
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch (labels): {magic}")

            labels = np.frombuffer(file.read(), dtype=np.uint8)

        # ---- Load images ----
        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch (images): {magic}")

            image_data = np.frombuffer(file.read(), dtype=np.uint8)

        images = image_data.reshape(size, rows, cols)

        return images, labels

    def load_data(
        self,
    ) -> Tuple[Tuple[UInt8Array, UInt8Array], Tuple[UInt8Array, UInt8Array]]:
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath,
            self.training_labels_filepath,
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath,
            self.test_labels_filepath,
        )

        return (x_train, y_train), (x_test, y_test)
 

def main():
    #
    # Verify Reading Dataset via MnistDataloader class
    #
    import random
    import matplotlib.pyplot as plt
    from pathlib import Path

    #
    # Set file paths based on added MNIST Datasets
    #

    ROOT = Path(__file__).resolve().parents[1]
    training_images_filepath = ROOT / 'dataset' / 'train-images-idx3-ubyte/train-images-idx3-ubyte'
    training_labels_filepath = ROOT / 'dataset' / 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    test_images_filepath = ROOT / 'dataset' / 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_filepath = ROOT / 'dataset' / 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

    #
    # Helper function to show a list of images with their relating titles
    #
    def show_images(images, title_texts):
        cols = 5
        rows = int(len(images)/cols) + 1
        plt.figure(figsize=(30,20))
        index = 1    
        for x in zip(images, title_texts):        
            image = x[0]        
            title_text = x[1]
            plt.subplot(rows, cols, index)        
            plt.imshow(image, cmap=plt.cm.gray)
            if (title_text != ''):
                plt.title(title_text, fontsize = 15);        
            index += 1
        plt.show()

    #
    # Load MINST dataset
    #
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    #
    # Show some random training and test images 
    #
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])        
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

    show_images(images_2_show, titles_2_show)

if __name__ == '__main__':
    main()