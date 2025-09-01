import pandas as pd
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

input_csv = "mnist_test.csv"
output_csv = "augmented_data.csv"

data = pd.read_csv(input_csv)
labels = data.iloc[:, 0].values
images = data.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.uint8)

def augment_image(image):
    pil_img = Image.fromarray(image)
    angle = random.uniform(-15, 15)
    pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    scale = random.uniform(0.85, 1.15)
    new_size = max(1, int(28 * scale))
    pil_img = pil_img.resize((new_size, new_size), Image.BILINEAR)
    result_img = Image.new("L", (28, 28), color=0)
    if new_size > 28:
        left = (new_size - 28) // 2
        top = (new_size - 28) // 2
        result_img = pil_img.crop((left, top, left + 28, top + 28))
    else:
        left = (28 - new_size) // 2
        top = (28 - new_size) // 2
        result_img.paste(pil_img, (left, top))
    shift_x = random.randint(-2, 2)
    shift_y = random.randint(-2, 2)
    shifted = Image.new("L", (28, 28), color=0)
    shifted.paste(result_img, (shift_x, shift_y))
    result = np.array(shifted)
    if random.random() > 0.5:
        noise = np.random.normal(0, 10, result.shape)
        result = np.clip(result + noise, 0, 255).astype(np.uint8)
    return result

all_data = []

for i, (label, image) in tqdm(enumerate(zip(labels, images))):
    if (i + 1) % 5 == 0:
        image = augment_image(image)
    all_data.append([int(label)] + image.flatten().tolist())
print(f"saving data to file {output_csv}")
column_names = ["label"] + [f"pixel{i}" for i in range(784)]
df = pd.DataFrame(all_data, columns=column_names)
df.to_csv(output_csv, index=False)
print("done")