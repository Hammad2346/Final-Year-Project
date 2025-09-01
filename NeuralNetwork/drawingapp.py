import tkinter as tk
from tkinter import Canvas, Button, Label, Frame
from PIL import Image, ImageDraw, ImageTk, ImageFilter
from trainednetwork import Network
import numpy as np

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition")
        self.root.configure(bg="#1e1e1e")
        self.root.attributes("-fullscreen", True)

        self.network = Network()
        self.canvasSize = 2100
        self.mnistSize = 28
        self.lineWidth = max(5, int(self.canvasSize * 0.075)) 

        self.canvasFrame = Frame(root, bg="#2d2d2d")
        self.canvasFrame.pack(side=tk.LEFT, padx=10, pady=30)

        self.canvasLabel = Label(self.canvasFrame, text="Draw a digit (0–9):", bg="#2d2d2d", fg="white", font=("Arial", 18))
        self.canvasLabel.pack()

        self.canvas = Canvas(self.canvasFrame, width=self.canvasSize, height=self.canvasSize, bg="black", bd=0, highlightthickness=0)
        self.canvas.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.onRelease)

        self.resultFrame = Frame(root, bg="#1e1e1e")
        self.resultFrame.pack(side=tk.RIGHT, padx=10, pady=30, fill=tk.Y)

        self.resultLabel = Label(self.resultFrame, text="Prediction:", bg="#1e1e1e", fg="white", font=("Arial", 20))
        self.resultLabel.pack()

        self.predictionLabel = Label(self.resultFrame, text="Draw something!", font=("Arial", 48), bg="#1e1e1e", fg="#00ddff")
        self.predictionLabel.pack(pady=30)

        self.probabilityLabels = []
        for i in range(10):
            probFrame = Frame(self.resultFrame, bg="#1e1e1e")
            probFrame.pack(fill=tk.X, pady=4)

            digitLabel = Label(probFrame, text=f"Digit {i}:", width=8, anchor="w", bg="#1e1e1e", fg="white", font=("Arial", 14))
            digitLabel.pack(side=tk.LEFT)

            probLabel = Label(probFrame, text="0%", width=10, anchor="w", bg="#1e1e1e", fg="white", font=("Arial", 14))
            probLabel.pack(side=tk.LEFT)

            barCanvas = Canvas(probFrame, width=200, height=10, bg="#333333", bd=0, highlightthickness=0)
            barCanvas.pack(side=tk.LEFT, padx=10)

            self.probabilityLabels.append((probLabel, barCanvas))

        self.buttonFrame = Frame(self.resultFrame, bg="#1e1e1e")
        self.buttonFrame.pack(pady=30)

        self.predictButton = Button(self.buttonFrame, text="Predict", command=self.predictDigit, bg="#00cc66", fg="white", font=("Arial", 14), padx=20, pady=8)
        self.predictButton.pack(side=tk.LEFT, padx=10)

        self.clearButton = Button(self.buttonFrame, text="Clear", command=self.clearCanvas, bg="#ff4444", fg="white", font=("Arial", 14), padx=20, pady=8)
        self.clearButton.pack(side=tk.LEFT, padx=10)

        self.exitButton = Button(self.buttonFrame, text="Exit", command=self.root.destroy, bg="#444444", fg="white", font=("Arial", 14), padx=20, pady=8)
        self.exitButton.pack(side=tk.LEFT, padx=10)

        self.image = Image.new("L", (self.canvasSize, self.canvasSize), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.oldX = None
        self.oldY = None

    def paint(self, event):
        x, y = event.x, event.y
        if self.oldX is not None and self.oldY is not None:
            self.canvas.create_line(self.oldX, self.oldY, x, y, width=self.lineWidth, fill="white", capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.oldX, self.oldY, x, y], fill="white", width=self.lineWidth)
        else:
            radius = self.lineWidth // 2
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white", outline="white")
            self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="white")
        self.oldX = x
        self.oldY = y

    def onRelease(self, event):
        self.oldX = None
        self.oldY = None

    def clearCanvas(self):
        self.canvas.delete("all")
        self.oldX = None
        self.oldY = None
        self.image = Image.new("L", (self.canvasSize, self.canvasSize), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.predictionLabel.config(text="Draw something!")
        for probLabel, barCanvas in self.probabilityLabels:
            probLabel.config(text="0%")
            barCanvas.delete("all")

    def preprocessImage(self):
        blurred = self.image.filter(ImageFilter.BoxBlur(1))
        mnistImage = blurred.convert("L")
        bbox = mnistImage.getbbox()
        if bbox is None:
            return np.zeros((self.mnistSize * self.mnistSize,), dtype=np.float32)
        digit = mnistImage.crop(bbox)
        maxDim = max(digit.size)
        scale = 20 / maxDim
        newSize = tuple([int(dim * scale) for dim in digit.size])
        digit = digit.resize(newSize, Image.LANCZOS)
        centered = Image.new("L", (28, 28), color=0)
        upperLeft = ((28 - newSize[0]) // 2, (28 - newSize[1]) // 2)
        centered.paste(digit, upperLeft)
        pixels = np.array(centered).astype(np.float32) / 255.0
        return pixels.flatten()

    def predictDigit(self):
        
        try:
            pixels = self.preprocessImage()
            prediction = self.network.predict(pixels)
            predictedDigit = np.argmax(prediction)
            self.predictionLabel.config(text=f"{predictedDigit}")
            for i, (probLabel, barCanvas) in enumerate(self.probabilityLabels):
                prob = prediction[i] * 100
                probLabel.config(text=f"{prob:.1f}%")
                barCanvas.delete("all")
                barWidth = int(prob * 2)
                if barWidth > 0:
                    barCanvas.create_rectangle(0, 0, barWidth, 10, fill="#00ccff" if i == predictedDigit else "#666666", outline="")
        except Exception as e:
            self.predictionLabel.config(text="Error")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
