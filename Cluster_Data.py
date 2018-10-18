import warnings
import Clustering
import nltk
warnings.simplefilter(action='ignore', category=FutureWarning)
nltk.download('maxent_ne_chunker')
nltk.download('words')
import Feature_Generation



#Dataset cleaned and PreProcessed
format = ["Deactivation of the finger protection causes the LED to be deactivated.",
          "When the central locking system is active and the vehicle is closed, the LED lights up.",
          "If the central locking system is inactive and the vehicle is not closed,the LED does not light up.",
          "Deactivating the central locking system also turns off the LED.",
          "When all windows are still, the LED does not light.",
          "When the button for closing a window is pressed and the corresponding window moves up,the LED indicates that a window closes.",
          "When the button for opening a window is pressed and the corresponding window moves down,the LED indicates that a window is opening.",
          "Closing of the opening or closing process of the windows leads to deactivation of the LED.",
          "The LED is lit when the exterior mirror is folded.",
          "The LED does not light up when the exterior mirror is folded out.",
          "Switching between the states of the viewing mirror leads to the change between the states of the LED.",
          "A special LED for each direction illuminates when the viewing mirror is in the maximum position in this direction.",
          "If a special LED is illuminated for each direction,when the exterior mirror is in the maximum position in this direction and the mirror is back, then When the exterior mirror heating is activated, the LED is lit.",
          "If the exterior mirror heating is deactivated,the LED does not light up.",
          "Deactivating the exterior mirror heating leads to the deactivation of the LED.",
          "Pressing the button to close the window will cause the window to close as soon as the print is exerted or the window has reached its top.",
          "Pressing the button to open the window will cause the window to open as long as the print is running or the window has reached the bottom.",
          "Pressing the button to open the window will open the window completely.",
          "Pressing the button to close the window will cause the window to close completely.",
          "If the window is just opened, pressing the close button will cause the window to stop.",
          "If the window is just opened, nothing happens by pressing the button.",
          "If the window is closed, press the button to open the window.",
          "If the window is closed, nothing happens by pressing the key.",
          "When the window is completely open, pressing the button for opening does not have any effect.",
          "If the window is completely closed, pressing the key for closing does not have any effect.",
          "As soon as the window is closed and an obstacle in the window frame is detected by back pressure,the finger protection is activated.",
          "If the anti-trap protection is activated, the window can no longer be closed.",
          "The finger protection can be deactivated by briefly pressing the open button.",
          "If the LED is present and the finger protection is activated, the LED is lit.",
          "If the LED is present and the finger protection is not activated, the LED does not light up.",
          "When the mirror is folded, pressing the central button will cause the mirror to fold out.",
          "When the mirror is folded out and the vehicle is stationary,pressing the central button causes the mirror to fold.",
          "When the mirror is unfolded and the vehicle is in motion,pressing the central button has no effect.",
          "If the LED is present and the mirror is folded in, the LED is lit.",
          "If the LED is present and the mirror is folded out, the LED does not light up.",
          "Pressing the directional button left of the electrical exterior mirror,causes the exterior mirror to move to the left.",
          "This only works when the exterior mirror has moving margin to the left.",
          "Pressing the directional button right of the electrical exterior mirror causes the exterior mirror to move to the right.",
          "This only works when the exterior mirror has movement margin to the right.",
          "Pressing the directional button up of the electrical exterior mirror causes the exterior mirror to move upwards.",
          "This only works when the exterior mirror has moving range up.",
          "Pressing on the directional button down of the electrical exterior mirror causes the exterior mirror to move downwards. ",
          "This only works when the exterior mirror has movement range down.",
          "Two temperatures are defined,a minimum temperature and a maximum temperature.",
          "If the temperature measurement on the exterior mirror falls below the minimum temperature,the exterior mirror heating is activated.",
          "Switching off the vehicle leads to the switch-off of the exterior mirror heater.",
          "When the exterior mirror heating is activated and the maximum temperature is exceeded,the exterior mirror heating is deactivated.",
          "If the LED is present and the exterior mirror heating is activated, the LED is lit.",
          "If the LED is present and the exterior mirror heater is not activated,the LED does not light up.",
          "Violent opening of a door causes an alarm to be triggered when the alarm is active",
          "Violent opening of a door does not trigger an alarm when the alarm system is deactivated.",
          "If an LED is present, an ongoing alarm is signaled by an LED.",
          "Alarm is automatically terminated after 20 seconds,after which a message will be displayed if LEDs are present.",
          "Alarm can be deactivated by unlocking the vehicle. ",
          "This does not lead to a message when LEDs are present.",
          "When recording a movement in the interior while the alarm system is activated, alarm is triggered.",
          "The interior monitoring is deactivated as soon as the alarm system is deactivated.",
          "If the LED for the alarm system is available, a special LED is lit when the interior monitoring is activated.",
          "If the LED for the alarm system is present and the interior monitoring is deactivated,the special LED does not light up.",
          "Pressing the locking button and inactive central locking activates the central locking system,which closes all doors.",
          "Pressing the locking button and active central locking system will not trigger any action.",
          "When pressing the unlocking button and active central locking system,the central locking system is deactivated, whereby all doors are unlocked.",
          "No action is triggered when the unlocking button and the inactive central locking system are pressed.",
          "By closing a door with inactive central locking, the central locking system is activated,whereby all doors are closed.",
          "By closing a door with the central locking system active,no action is triggered.",
          "By unlocking a door with active central locking system,the central locking system is deactivated, whereby all doors are unlocked.",
          "By unlocking a door with inactive central locking system, no action is triggered.",
          "When the window is locked, when the central locking system is inactive and the automatic power window is in operation, all windows are closed automatically,if they are open,otherwise they will be blocked.",
          "When unlocking the window, active central locking system and the existence of the automatic power window, the blocking of all windows is canceled.",
          "When the window is locked, if the central locking system is inactive and the manual power window is activated, all windows are blocked.",
          "When unlocking the window, active central locking system and the existence of the manual power window, the blocking of all windows is canceled.",
          "When a defined speed is exceeded, all doors are closed when they are not closed.",
          "When a defined speed is exceeded, nothing happens if the doors have already been completed manually.",
          "If the speed is reduced so that it falls below the defined value and the doors have been closed by the automatic locking feature, the doors are unlocked.",
          "If the speed is reduced so that it falls below the defined value and the doors have not been manually closed by the automatic locking feature, the doors are not unlocked.",
          "Pressing the remote control button to lock activates the central locking system if the central locking system is not activated.",
          "Pressing the remote control button to unlock unlocks the central locking system when the central locking system is activated.",
          "Pressing the remote control button for locking has no effect when the central locking is active.",
          "Pressure on the remote control button for unlocking has no effect if the central locking is decanted.",
          "By pressing the corresponding button on the remote control,the alarm system is activated when it was previously inactive.",
          "If the alarm is active, pressing the remote control button does not have any effect.",
          "By pressing the corresponding button on the remote control,the alarm system is deactivated if it was active before.",
          "If the alarm is not active, pressing the remote control button does not have any effect.",
          "The passage of ten seconds between the unlocking and the opening of a door leads to the door being closed again.",
          "The unlocking and opening of a door within 10 seconds causes the door to not be automatically closed again.",
          "Pressing the remote control button to open the window causes the window to open for opening.",
          "Pressing the remote control button to close the window will allow the window to close up for closing.",
          "If the window is being opened, pressing the remote control button will close the window.",
          "If the window is just opened, nothing happens by pressing the button.",
          "If the window is closed, press the remote control button to open the window.",
          "If the window is closed, nothing happens by pressing the key.",
          "If the window is fully open, pressing the remote control button will have no effect.",
          "If the window is completely closed, the pressing of the remote control button for closing does not have any effect."]




#KMeans Clustering Algorithm has been Run on the PreProcessed data.

# print('===============First clustering algorithm===================')
# kmeans = Clustering.cluster_texts(format,clusters=20)
# Clustering.pprint(dict(kmeans))
# print('===============First clustering algorithm===================')

print('===============Second clustering algorithm===================')
aggo = Clustering.cluster_texts(format,clusters=12)
Clustering.pprint(dict(aggo))
print('===============Second clustering algorithm===================')

#Named-Entity tags and POS-Tags has been made use of to derive the results.
clus = []
a=dict(kmeans)
for i,j in zip(a.values(),a.keys()):
    print('----------------cluster' + ' ' + str(j) + '-------------')
    for value in i:
        clus.append(format[value])
    Feature_Generation.named_Entity(""" """.join(clus))
    clus = []
    print('-------------------cluster'+' '+ str(j) + '-------------')