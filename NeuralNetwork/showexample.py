import pandas as pd
import tkinter as tk
from tkinter import Canvas, Button, Label, Frame
from PIL import Image, ImageDraw, ImageTk
import random
from trainednetwork import Network
import numpy as np


df = pd.read_csv("NeuralNetwork/mnist_test.csv")
network = Network()
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                       'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b','c', 'd', 'e', 
                       'f', 'g', 'h','i','j','k','l','m', 'n','o','p', 'q', 
                       'r', 's', 't','u','v','w','x','y','z']
root = tk.Tk()
root.title("MNIST Digit Viewer")
root.configure(bg="black")
root.attributes("-zoomed", True) 

frame = Frame(root, bg="black")
frame.pack(expand=True)

canvas_size = 1000
photoCanvas = Canvas(frame, width=canvas_size, height=canvas_size, bg="black", highlightthickness=0)
photoCanvas.pack(pady=30)

predictionLabel = Label(frame, text="Prediction: _", font=("Helvetica", 24), fg="white", bg="black")
predictionLabel.pack(pady=10)

originalLabel = Label(frame, text="Actual: _", font=("Helvetica", 24), fg="white", bg="black")
originalLabel.pack(pady=10)

image_ref = None


def randomExample():
    global image_ref
    example = df.iloc[random.randint(0, len(df) - 1)]
    label = example.iloc[0]
    raw_pixels = example[1:].values.astype(np.uint8).reshape(28, 28)
    
    prediction = np.argmax(network.predict(raw_pixels.flatten()))

    img = Image.fromarray(raw_pixels).resize((canvas_size, canvas_size), Image.NEAREST).convert("L")
    image_ref = ImageTk.PhotoImage(img)

    photoCanvas.delete("all")
    photoCanvas.create_image(0, 0, anchor="nw", image=image_ref)

    predictionLabel.config(text=f"Prediction: {classes[prediction]}")
    originalLabel.config(text=f"Actual: {classes[int(label)]}")

randomButton = Button(
    frame,
    text="Show Random Digit",
    command=randomExample,
    font=("Helvetica", 18),
    bg="gray15",
    fg="white",
    activebackground="gray30",
    width=20,
    height=2
)
randomButton.pack(pady=30)

root.mainloop()
