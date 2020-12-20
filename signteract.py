import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tkcalendar import Calendar
from datetime import datetime
import tkinter as tk
from tkinter import *
import cv2
import numpy as np
import pyttsx3



#list of alphabets that are identified
ALPHASTR = ["A", "C", "D", "H", "J", "R", "T", "U", "W"]

#variable for the cv2 window
ORG = (200, 125)
COLOR = (255,255,255)
THICKNESS = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX 
FONTSCALE = 5


class Alarm:
    """ class that displays an alarm widget """
    
    def __init__(self, root, engine):
        """ 
            construct an alarm widget with the parent root 
        
            Args:
                root: tk.Tk()
                engine: pyttsx3.engine
            
            Returns:
                None
        
        """        
        #speech engine
        self.engine = engine
        
        #values for time
        self.values = ["AM", "PM"]
        
        #main root window
        self.root = root
        self.root.title("Signteract Alarm")
        self.root.config(bg = "azure")
        
        #title label
        self.title = Label(self.root, text = "Set an Alarm")
        self.title.config(font = ("Comic Sans MS Bold Italic", 15), bg="azure")
        
        #calendar title
        self.cal_title = Label(self.root, text="Set the date")
        self.cal_title.config(font = ("Comic Sans MS Bold Italic", 12), bg="azure")
        
        #calendar frame
        self.cal_frame = Frame(self.root)
        self.cal_frame.configure(bg= "azure")
        
        #calendar
        self.cal = Calendar(self.cal_frame,
                       font="Arial 14", selectmode='day',
                       cursor="hand1", year=2020, month=12, day=19)
        
        #time title
        self.time_title = Label(self.root, text="Set the time")
        self.time_title.config(font=("Comic Sans MS Bold Italic", 12), bg="azure")
        
        #time frame
        self.time_frame = Frame(self.root)
        self.time_frame.configure(bg= "azure")
        
        #time labels
        self.hr_label = Label(self.time_frame, text = "Hours") 
        self.min_label = Label(self.time_frame, text = "Minutes")        
        self.hr_label.config(font=("Comic Sans MS Italic", 10), bg="azure")
        self.min_label.config(font=("Comic Sans MS Italic", 10), bg="azure")
        
        #time strings
        self.min_str = tk.StringVar(self.time_frame, '30')
        self.hr_str = tk.StringVar(self.time_frame, '2')
        self.ampm_str = tk.StringVar(self.time_frame, 'AM')
        
        #time spinbox
        self.hr_spbox = tk.Spinbox(self.time_frame, from_=0, to=12, wrap = True,\
                                   textvariable=self.hr_str, width=5, state="readonly")
        self.min_spbox = tk.Spinbox(self.time_frame, from_=0, to=59, wrap = True, \
                                    textvariable=self.min_str, width=5, state="readonly")
        self.ampm_spbox = tk.Spinbox(self.time_frame, values=self.values, wrap = True, \
                                     textvariable=self.ampm_str, width=5, state="readonly")
            
        self.hr_spbox.config(font=("Comic Sans MS Bold Italic", 10))
        self.min_spbox.config(font=("Comic Sans MS Bold Italic", 10))
        self.ampm_spbox.config(font=("Comic Sans MS Bold Italic", 10))
            
    
        #set alarm button
        self.btn = Button(self.time_frame, text="SET ALARM", command=self.get_alarm_time)
        
        
        #building the UI
        self.title.pack(expand = True)
        self.cal_title.pack(expand =True)
        self.cal_frame.pack(expand=True)
        self.time_title.pack(expand=True)
        self.time_frame.pack(expand=True)
        
        #building the calendar frame
        self.cal.pack(expand=True)
        
        #building the time frame
        self.hr_label.grid(row = 2, column = 1, sticky = E, pady = 2) 
        self.min_label.grid(row = 2, column = 2, sticky = E, pady = 2)  
        self.hr_spbox.grid(row = 3, column = 1, sticky=E, pady=2, padx=5)
        self.min_spbox.grid(row = 3, column = 2, sticky=E, pady=2, padx=5)
        self.ampm_spbox.grid(row=3, column=3, sticky=E, pady=2, padx=5)
        
        #adding the button
        self.btn.grid(row=5, column = 1, columnspan=2, sticky=E, pady=2)
        
    def get_alarm_time(self):
        
        """ gets the alarm time from the widget and converts it to speech """
        
        #getting the values from the UI
        mins = self.min_str.get()
        hrs = self.hr_str.get()
        date = self.cal.selection_get()
        ampm = self.ampm_str.get()
        
        #getting the date and month
        month = date.strftime("%B")
        date = date.strftime("%d")
        
        #construction of the command
        command = "Set an alarm on "+date +month+" at "+hrs+" "+mins+" "+ampm
        
        #saying the command
        self.engine.say("Alexa")
        self.engine.runAndWait()
        self.engine.say(command)
        self.engine.runAndWait()
        
        #destroying the window
        self.root.destroy()
        LaunchWindow(tk.Tk(), self.engine)
        
class Timer:
    """ a class that creates a timer widget """
    def __init__(self, root, engine):
        """ 
            construct a timer widget with the parent root 
        
            Args:
                root: tk.Tk()
                engine: pyttsx3.engine
            
            Returns:
                None
        
        """
        #setting up the engine
        self.engine= engine
        
        #setting up the root window
        self.root = root
        self.root.title("Signteract Timer")
        self.root.configure(bg="azure")
        
        #time frame
        self.time_frame= Frame(self.root)
        self.time_frame.configure(bg="azure")
        
        #time title
        self.time_title = Label(self.time_frame, text="SET THE TIMER")
        self.time_title.config(font=("Comic Sans MS Bold Italic", 12), bg = "azure")
        
        #string values to store the timer values
        self.min_str = tk.StringVar(self.time_frame, '0')
        self.hr_str = tk.StringVar(self.time_frame, '0')
        self.sec_str = tk.StringVar(self.time_frame, '0')
        
        #time labels
        self.hr_label = Label(self.time_frame, text = "Hours") 
        self.min_label = Label(self.time_frame, text = "Minutes")
        self.sec_label = Label(self.time_frame, text = "Seconds")
        self.hr_label.config(font=("Comic Sans MS Bold Italic", 10), bg="azure")
        self.min_label.config(font=("Comic Sans MS Bold Italic", 10), bg="azure")
        self.sec_label.config(font=("Comic Sans MS Bold Italic", 10), bg="azure")
        
        #spinboxes for the time values
        self.hr_spbox = tk.Spinbox(self.time_frame, from_=0, to=12, wrap = True,\
                                   textvariable=self.hr_str, width=5, state="readonly")
        self.min_spbox = tk.Spinbox(self.time_frame, from_=0, to=59, wrap = True,\
                                    textvariable=self.min_str, width=5, state="readonly")
        self.sec_spbox = tk.Spinbox(self.time_frame, from_ = 0,to = 59, wrap = True,\
                                    textvariable=self.sec_str, width=5, state="readonly")
        self.hr_spbox.config(font=("Comic Sans MS Bold Italic", 10))
        self.min_spbox.config(font=("Comic Sans MS Bold Italic", 10))
        self.sec_spbox.config(font=("Comic Sans MS Bold Italic", 10))
    
    
        #set button        
        self.btn = Button(self.time_frame, text="Set Timer", command=self.get_timer_value)
        
        #building the UI
        self.time_title.pack(expand=True)
        self.time_frame.pack(expand=True)
        self.time_title.grid(row=1, column=1, columnspan=3)
        
        self.hr_label.grid(row = 2, column = 1, sticky = E, pady = 2) 
        self.min_label.grid(row = 2, column = 2, sticky = E, pady = 2)  
        self.sec_label.grid(row = 2, column = 3, sticky = E, pady = 2)  
        
        self.hr_spbox.grid(row = 3, column = 1, sticky=E, pady=2, padx=3)
        self.min_spbox.grid(row = 3, column = 2, sticky=E, pady=2, padx=3)
        self.sec_spbox.grid(row=3, column=3, sticky=E, pady=2, padx=3)
        self.btn.grid(row=5, column = 1, columnspan=2, sticky=E, pady=5)
        
        
    def get_timer_value(self):
        hrs = self.hr_str.get()
        mins = self.min_str.get()
        sec = self.sec_str.get()
        
        command = "Alexa set a timer for "
        
        if int(hrs)>0: 
            command+=" "+str(hrs)+" hours"
            
        if int(mins)>0:
            command+= " and "+str(mins)+" minutes"
            
        if int(sec)>0: 
            command+= " and "+str(sec)+" seconds"
            
        #running the command
        self.engine.say("Alexa")
        self.engine.runAndWait()
        self.engine.say(command)
        self.engine.runAndWait()
        
        #destroying the UI window
        self.root.destroy()
        LaunchWindow(tk.Tk(), self.engine)




class LaunchWindow:
    """ 
        creates the home window of the Signteract application 
        
        Args:
            root: tk.Tk()
            engine: pyttsx3 engine
    
    """
    def __init__(self,root, engine):
    
        self.root = root    
        self.root.title("Signteract")
        
        #setting the speech engine
        self.engine = engine

        self.frame = tk.Frame(root, height=750, width=750, bg="azure")
        self.frame.pack()
        
        self.logo = tk.PhotoImage(file="logo_2.png")
        self.logo_label = tk.Label(self.frame, image=self.logo, bg="azure")
        self.logo_label.place(x=325,y=90)

        self.mainTitle = tk.Label(self.frame, text="Signteract", fg="purple" ,\
                                  bg="azure", font=("Comic Sans", 50, \
                                  "bold")).place( x=200, y=200)
            
        self.tagLine = tk.Label(self.frame, text="Sign and Interact", fg="purple" ,\
                                  bg="azure", font=("Comic Sans", 18, \
                                  "bold italic")).place( x=270, y=295)    
        

       
        self.btn = tk.Button(self.frame, text="Start Signteraction", width=18, height=1, \
                             fg="black", bg="DeepSkyBlue2",font=("Comic Sans", 15),\
                             command=self.nextWindow)
        self.btn.place(x=260,y=450)
        

          
    def nextWindow(self):
        """ clears the widgets in the root screen and takes to the next window """
        
        for item in self.root.winfo_children():
          #deleting the widget from the screen
          item.pack_forget()
          
        #open the next window
        CommandWindow(self.root, self.engine)
        
        
        
class CommandWindow:
    """ creates the second screen of the UI """
    
    def __init__(self,root, engine):
        """
        initializes the UI widgets

        Parameters
        ----------
        root : tk.Tk()
        engine: pyttsx3 engine
            
        Returns
        -------
        None.

        """
        
        #customizing the root widget        
        self.root = root
        self.root.title("Signteract")
        
        #setting the speech engine
        self.engine = engine
        
        #available commands
        self.commands_dict = {"A": "Set an alarm", 
                         "C": "Confirm command", 
                         "D": "Volume down", 
                         "H": "Hi", 
                         "J": "Tell me a joke", 
                         "R": "Redo the command",
                         "T": "Set a timer", 
                         "U": "Volume up",
                         "W": "What's the weather"}
        
        
        #variable to store the predicted text
        self.predicted_text = tk.StringVar()
        self.comand_msg = tk.StringVar()
        self.conf_msg = tk.StringVar(value= "Confirmation Message")
        
        #instantiating the model
        self.model = self.load_model("asl_edges_extended.h5")
        
        #starting the signing window
        self.new_command()

        #creating the UI for the interacting with the user
        self.frame = tk.Frame(root, height=750, width=750, bg="azure")
        self.frame.pack()
        
        #confirmation message label
        self.msg1 = tk.Label(self.frame, text="Show 'C' to confirm and 'R' to Redo",\
                             fg="black",  bg="azure",\
                             font=("Times New Roman", 18,"bold italic"))
        self.msg1.place(x=200,y=75)
        
        self.msg2 = tk.Label(self.frame, text="Detected Sign is",fg="black",  bg="azure",\
                             font=("Times New Roman", 18, "bold italic"))
        self.msg2.place(x=300,y=200)
        
        
        #predicted text label
        self.predicted_label = tk.Label(self.frame, textvariable=self.predicted_text,\
                                        fg="blue",  bg="azure", \
                                        font=("Times New Roman", 60, "bold"))
        self.predicted_label.place(x=350, y=250)
        
        
        #command label
        self.comand_label = tk.Label(self.frame, textvariable=self.comand_msg, height=2 ,\
                                   width=30 ,fg="black",  bg="pink", \
                                   font=("Times New Roman", 18, "bold italic"))
        self.comand_label.place(x=150,y=425)
        
        #confirmation message label
        self.conf_label = tk.Label(self.frame, textvariable=self.conf_msg, height=2 ,\
                                   width=30 ,fg="black",  bg="pink", \
                                   font=("Times New Roman", 18, "bold italic"))
        self.conf_label.place(x=150,y=500)
        
        
    
        
    def preprocess_img(self,img: np.array)-> np.array:
        """
        A function to preprocess the RGB image
        
        This function resizes and blurs the image and applies the Canny edge
        detection algorithm. It then converts the image to RGB and returns it.
    
        Parameters
        ----------
        img : np.array
            A numpy array representation of the RGB image .
    
        Returns
        -------
        A numpy array of the preprocessed image
    
        """
        #resizing the image
        img = cv2.resize(img, (299,299))
        
        #adding a blur
        img = cv2.blur(img, (5,5), cv2.BORDER_DEFAULT)
        
        img = cv2.Canny(img, 40, 110)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        return img
    

    def load_model(self,fname: str) -> tf.keras.Model:
        """
        A function to instantiate the deep learning model
        
        Args:
        fname : str
            Filename of the model to be loaded.
    
        Returns
        -------
        None.
    
        """
                    
        model = tf.keras.Sequential()
    
        model.add(tf.keras.layers.Input(shape = (299,299,3)))
        model.add(Conv2D(64, kernel_size=(5,5), strides=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
    
        model.add(Conv2D(128, kernel_size=(3,3), strides=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    
        model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(9, activation='softmax'))
    
        model.load_weights(fname)
        return model
    
      
    def start_video(self):
        """ starts the video camera and captures the frame """
        
        #starting the video capture
        vid = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
        
        #counter variable for the countdown
        j= 60
        
        while True:
            
            #capture the frame
            ret, frame = vid.read()
            #plot the signing window
            cv2.rectangle(frame, (50,150), (350,450), (0,255,0), 0)
            #countdown text
            cv2.putText(frame, str(j//20),ORG,FONT,FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA )
                
            #displaying the live feed
            cv2.imshow("live", frame)
            cv2.waitKey(1)
            
            #if the countdown is at 0 then destroy the windows
            if j==0:
                vid.release()
                cv2.destroyAllWindows()
                return frame
                
            
            j-=1
            
            
    def process_frame(self,frame):
        """ 
            A funtion to process the frame to find the symbol
            
            Args:
                frame: np.array()
            
            Returns:
                A string which is the predicted symbol   
        
        """
        #extracting the signing window
        roi = frame[150:450, 50:350]
            
        #preprocessing the image
        roi = self.preprocess_img(roi)
    
        #displaying the preprocessed image
        #cv2.imshow("hand", roi)
                
        re_roi = np.reshape(roi, (1,299,299,3))
        
        #predicting the sign from the image
        pred_class = np.argmax(self.model.predict(re_roi))
        
        #getting the letter
        pred_text = ALPHASTR[pred_class]
        
        return pred_text
    

    def confirm_command(self):
        """ verifies the prediction and says the command """
        
        #for confirming the sign, restart the video capture and process the frame
        conf_frame = self.start_video()
        conf_pred = self.process_frame(conf_frame)
        
        
    
        
        #if the symbol is R, then it means that the sign has been wrongly predicted
        #so the process is repeated and the new_command function is called
        
        
        if conf_pred == 'R':
            self.conf_msg.set("Wrongly predicted. Redo")
            #self.conf_label.pack()
            self.root.after(2000, self.create_new_window)
            
                     
        else:
            #clear the UI
            for item in self.root.winfo_children():
                    item.pack_forget()
                
            if self.pred == "A":
                Alarm(self.root,self.engine)
            elif self.pred== "T":
                Timer(self.root,self.engine)
            else:
                #make the speech engine say the command
                command = self.commands_dict[self.pred]
                
                self.engine.say("Alexa")
                self.engine.runAndWait()
                
                self.engine.say(command)
                self.engine.runAndWait()
                LaunchWindow(self.root, self.engine)
            
            
    def create_new_window(self):
        for item in self.root.winfo_children():
            item.pack_forget()
            
        CommandWindow(self.root, self.engine)
        
        
    def new_command(self):
        """ function to start the video, process it and get the symbol """
        
        #getting the frame from the video
        frame = self.start_video()
        
        #getting the prediction
        self.pred= self.process_frame(frame)
        
        #setting the label
        self.predicted_text.set(str(self.pred))
        
        #getting the command for the letter
        command = self.commands_dict[self.pred]
        
        #text to be displayed to the user
        assistant_dialog = "Is your command: "+command+"?\nIf yes, show C, else show R"
        
        #displaying the confirmation message to the user
        self.comand_msg.set(assistant_dialog)
        
        #confirming the command
        self.root.after(2000,self.confirm_command)
        

    
            
if __name__=="__main__":

    root = tk.Tk()
    root.resizable(0, 0)
    root.geometry("750x600")
    
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    
    launch_win = LaunchWindow(root, engine)
    root.mainloop()
    
    engine.stop()
    
