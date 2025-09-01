import pandas as pd
from PIL import Image
import numpy as np

data = pd.read_csv("emnist-balanced-test.csv", header=None)

labels = data.iloc[:, 0]
images = data.iloc[:, 1:]

savefile = "emnist_data.csv"

def correctImage(image):
    image = image.transpose(Image.ROTATE_90)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

if __name__ == "__main__":
    fixed_data = []

    for i in range(len(images)):
        img_array = images.iloc[i].values.astype(np.uint8).reshape(28, 28)
        pil_image = Image.fromarray(img_array)
        corrected_image = correctImage(pil_image)
        pixels_array = np.asarray(corrected_image).flatten()
        row = [labels.iloc[i]] + pixels_array.tolist()
        fixed_data.append(row)
        print(f"image {i} done")

    df_corrected = pd.DataFrame(fixed_data)
    df_corrected.to_csv(savefile, index=False)
