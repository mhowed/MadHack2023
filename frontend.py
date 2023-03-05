import customtkinter
import tkinter as tk
import time
import predict
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import yfinance as yahoo
import os.path
import absl.logging



class PredictFrontEnd(customtkinter.CTkFrame):
    def __init__(self, master=None):
        # Set the color theme
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")

        # Call const for customtkinter
        #super().__init__(master)
        self.master = master
        #self.pack()
        self.root = master
        # Make the layout
        self.make_widgets()
        self.root.geometry("1100x600")
        self.root.title("Stock Prediction")

        self.left_label = None
        self.right_label = None

        self.text = ""
    
    def make_widgets(self):
        # Create entry widget for ticker
        self.ticker_entry = customtkinter.CTkEntry(self.master, placeholder_text="Enter Ticker", bg_color='#1a1a1a', fg_color="#1a1a1a", border_color='#1a1a1a', placeholder_text_color='#857d7d', text_color='#70476b', font=('PT Mono', 14))
        self.ticker_entry.place(relx=0.5, rely=0.47, anchor=customtkinter.CENTER)
        self.ticker_entry.bind("<Return>", self.on_return)
        
        # Button that will leave prolly
        #self.button = customtkinter.CTkButton(master=root, text="Predict", text_color='#1a1a1a', height=10 , width=10, font=('PT Mono', 12), fg_color='#70476b', hover_color='#4f344c')
        #self.button.place(relx=0.5, rely=0.62, anchor=customtkinter.CENTER)
        # Create Model Accurracy Slider
        self.slider = customtkinter.CTkSlider(master=self.root, from_=0, to=10, button_corner_radius=2, corner_radius=2, border_width=1, progress_color='#70476b', border_color='#1a1a1a', fg_color='#1a1a1a', bg_color='#1a1a1a', height=30, width=300, button_length=0, button_color='#70476b', button_hover_color='#70476b')
        self.slider.place(relx=0.5, rely=0.545, anchor=customtkinter.CENTER)

        self.canvas = customtkinter.CTkCanvas(self.master, width= 50, height= 50)
        # Create a Label to Show Left slider value
        #self.left_label = customtkinter.CTkLabel(self.master, text_color="#70476b", text="Fastest")
        #self.left_label.place(relx=0.36, rely=0.57)
        #Create a label to show right slider value
       # self.right_label = customtkinter.CTkLabel(self.master, text_color="#70476b", text='Most Accurate')
        #self.right_label.place(relx=0.59, rely=0.57)


    def on_return(self, event):
        # Get the value from ticker_entry
        ticker = self.ticker_entry.get()
        self.accuracy_val = self.slider.get()
        self.slider.place_forget()
        self.ticker_entry.place_forget()
        print(ticker)
        # Change the count variable so the annimation stops
        self.ticker_entry.delete(0, customtkinter.END)        
        
        useOldModel = False
        if self.accuracy_val <= 5:
            self.accuracy_val = True
        else:
            self.accuracy_val = False
        print(self.accuracy_val)
        # disable more input
        #self.ticker_entry.configure(state='disabled', disabledforeground='black', disabledbackground='white')
        #self.ticker_entry.insert(0, "Disabled Text")
        #restart the annimation for waiting
        #print(type(self.slider), type(self.left_label))
        #self.animate_loading(0)
        #self.right_label.place_forget()
        #self.left_label.place_forget()
        

        time.sleep(1.5)
        
        train, valid, predictions, ticker, model = predict.prediction(ticker, useOldModel)

        x = valid.head(1)
        real = x["Close"].values
        pred = x["Predictions"].values
        print("Real / Pred:", real/pred)
        variance = 0
        if self.accuracy_val == True:
            variance = .1
        else:
            variance = .02 

        while (real/pred) > 1 + variance or (real/pred) < 1 - variance: #and not useOldModel:
            train, valid, predictions, ticker, model = predict.prediction(ticker)
            x = valid.head(1)
            real = x["Close"].values
            pred = x["Predictions"].values
            print("Real / Pred:", real/pred)

        model.save(f"data/{ticker}")
        print("Prediction for the next close:", str(predictions[-1])[1:-1])
        self.prediction = customtkinter.CTkLabel(self.master, text_color="#70476b", text=str(predictions[-1])[1:-1])
        self.prediction.place(relx=.50, rely=.50)

        #self.canvas.place(relx=.50, rely=.50, anchor=customtkinter.CENTER)
        #img = f"images/{ticker}.png"
        #self.canvas.create_image(20, 20, anchor=customtkinter.CENTER, image=img)

        return ticker, self.accuracy_val

    
    def animate_loading(self, count):
        print("animate_loading called")
        if count < 3:
            self.ticker_entry.delete(0, customtkinter.END)
            self.ticker_entry.insert(0, f"Loading {self.text}{'.'*(count+1)}")
            self.master.after(500, self.animate_loading, count+1)
            self.master.update()
        else:
            #TODO
            pass

def main():
    # Create the main scene
    root = customtkinter.CTk()
    predict_app = PredictFrontEnd(master=root)
    root.mainloop()

main()
