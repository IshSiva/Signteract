# Signteract
Sign And Interact with your Assistant: An AI translator to interact with your virtual assistant using ASL.

## Commands
This translator will help people with speaking and hearing difficulties communicate easily with personal assistants like Google Assistant, Amazon Alexa, Apple Siri etc.
Presently it has been designed for Alexa, so the commands invoke alexa in the beginning. For other assistants, update the wakeup command as required. <br>
Signteract recognizes 9 letters currently. They are <br>
A -> Set an alarm. <br>
C -> Confirm Command <br>
D -> Volume Down <br>
H -> Hi <br>
J -> Tell me a joke <br>
R -> Redo Command <br>
T -> Set a timer <br>
U -> Volume up <br>
W -> What's the weather <br>
Alarm and Timer opens a new window to get the date and time. <br>
Confirm and Redo commands are used to check the predictions. If the sign is wrongly identified then the user can show the Redo command and the command can be given again.

## Working
<ol>
  <li>Start the app </li>
  <li>Show the command </li>
  <li>If the command is correct then show C, else show R. </li>
  <li>If your command is "set an alarm", then set the date and time. If your command is "set a timer", then specify the time </li>
 </ol>
  

## Requirements
<ul>
  <li>Tensorflow - 2.3</li>
  <li>Keras - 2.4.3 </li>
  <li>OpenCV - 4.4.0.46 </li>
  <li>Pyttsx3 - 2.9 </li>
  <li>Tkinter - 8.6 </li>
  <li>TkCalendar - 1.6.1 </li>
 </ul>

## Dataset
The Deep Learning model was trained using the ASL dataset from Kaggle: https://www.kaggle.com/grassknoted/asl-alphabet
