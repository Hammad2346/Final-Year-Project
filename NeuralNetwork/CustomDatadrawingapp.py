import tkinter as tk
from tkinter import Canvas, Button, Label, Frame
from PIL import Image, ImageDraw,ImageFilter
import numpy as np
import random
import csv
import os


class Gui:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Character")
        self.root.configure(bg="#1e1e1e")
        self.root.attributes("-fullscreen", True)
        self.imagePixels=28


        self.randomint=0
        self.classes= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                       'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b','c', 'd', 'e', 
                       'f', 'g', 'h','i','j','k','l','m', 'n','o','p', 'q', 
                       'r', 's', 't','u','v','w','x','y','z']
        self.classeslen=len(self.classes)

        self.filePath="customdata62.csv"


        self.canvasSize = 600
        self.lineWidth = max(5, int(self.canvasSize * 0.075))
        self.charToDraw = self.classes[self.randomint]

        # Canvas Frame
        self.canvasFrame = Frame(root, bg="#2d2d2d")
        self.canvasFrame.pack(side=tk.LEFT, padx=20, pady=20)

        self.label = Label(self.canvasFrame, text=f"Draw a '{self.charToDraw}'", font=("Arial", 20), bg="#2d2d2d", fg="white")
        self.label.pack(pady=(0, 10))

        self.canvas = Canvas(self.canvasFrame, width=self.canvasSize, height=self.canvasSize, bg="black", bd=0, highlightthickness=0)
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.onRelease)

     
        self.buttonFrame = Frame(root, bg="#1e1e1e")
        self.buttonFrame.pack( fill=tk.X, padx=20, pady=20)

        self.saveButton = Button(self.buttonFrame, text="Save", command=self.saveImage, bg="#00cc66", fg="white", font=("Arial", 16), padx=50, pady=20)
        self.saveButton.pack(side=tk.LEFT,padx=10,pady=10)

        self.clearButton = Button(self.buttonFrame, text="Clear", command=self.clearCanvas, bg="#ff4444", fg="white", font=("Arial", 16), padx=20, pady=10)
        self.clearButton.pack(side=tk.LEFT,padx=10,pady=10)

        self.exitButton = Button(self.buttonFrame, text="Exit", command=self.root.destroy, bg="#444444", fg="white", font=("Arial", 16), padx=20, pady=10)
        self.exitButton.pack(pady=10)

        
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
            r = self.lineWidth // 2
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill="white")
        self.oldX = x
        self.oldY = y

    def onRelease(self, event):
        self.oldX = None
        self.oldY = None

    def clearCanvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvasSize, self.canvasSize), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def saveImage(self):
        pixels = self.processImage()
        label = self.randomint
        dataRow = [label] + pixels.tolist()

        try:
            fileExists = os.path.isfile(self.filePath)
            
            with open(self.filePath, mode="a", newline="") as file:
                writer = csv.writer(file)
                
                if not fileExists:
                    header = ["label"] + [f"pixel{i}" for i in range(len(pixels))]
                    writer.writerow(header)
                
                writer.writerow(dataRow)  

            

            self.getRandomCharacter()
            self.charToDraw = self.classes[self.randomint]
            self.label.config(text=f"Draw a '{self.charToDraw}'")
            self.clearCanvas()

        except Exception as e:
            print(f"Save failed: {e}")






    def getRandomCharacter(self):
        self.randomint=random.randint(0,self.classeslen)


    def processImage(self):
        blurred=self.image.filter(ImageFilter.BoxBlur(1))
        grayscaled=blurred.convert('L')
        resized = grayscaled.resize((28, 28), resample=Image.LANCZOS)



        # boundingbox=grayscaled.getbbox()
        # if boundingbox is None:
        #     return np.zeros((self.imagePixels,self.imagePixels),dtype=np.float32)
        
        # character=grayscaled.crop(boundingbox)

        # maxDim=max(character.size)

        # scale=20/maxDim

        # newsize= [int(dim*scale) for dim in character.size]

        # resizedCharacter=character.resize(newsize)

        # backgroungImage=Image.new(mode="L",size=(self.imagePixels,self.imagePixels), color=0)

        # topleft= ((28-newsize[0])//2,(28-newsize[1])//2)

        # fullImage=backgroungImage.paste(resizedCharacter,topleft)

        pixels=np.array(resized).astype(dtype=np.float32)

        return pixels.flatten()
        






if __name__ == "__main__":
    root = tk.Tk()
    app = Gui(root)
    root.mainloop()
