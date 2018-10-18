from keyword_extractor import KeywordExtractor
import gensim
import warnings
import gensim.downloader as api
import warnings
import clustering
warnings.simplefilter(action='ignore', category=FutureWarning)
# import h5py
# warnings.resetwarnings()
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

text = "When the alarm system is activated, this is indicated by the light of the LED." \
                    "If the alarm is deactivated, the LED does not light up." \
                    "When an alarm is triggered, this is signaled by the light of the LED." \
                    "If no alarm is triggered, the LED does not light up." \
                    "When the interior monitoring is activated, this is signaled by the light of the LED." \
                    "If the indoor monitoring is not activated, the LED does not light up." \
                    "The LED, which signals the triggered alarm, can only be reset by pressing the reset button,which is next to the LED. " \
                    "The key has only one effect when the ignition key is in the ignition." \
                    "If the finger protection has been tripped and the electric window lifter is deactivated, the LED is lit." \
                    "If the finger protection has not been triggered, the LED does not light up." \
                    "Deactivation of the finger protection causes the LED to be deactivated." \
                    "When the central locking system is active and the vehicle is closed, the LED lights up." \
                    "If the central locking system is inactive and the vehicle is not closed,the LED does not light up." \
                    "Deactivating the central locking system also turns off the LED." \
                    "When all windows are still, the LED does not light." \
                    "When the button for closing a window is pressed and the corresponding window moves up,the LED indicates that a window closes." \
                    "When the button for opening a window is pressed and the corresponding window moves down,the LED indicates that a window is opening." \
                    "Closing of the opening or closing process of the windows leads to deactivation of the LED." \
                    "The LED is lit when the exterior mirror is folded." \
                    "The LED does not light up when the exterior mirror is folded out." \
                    "Switching between the states of the viewing mirror leads to the change between the states of the LED." \
                    "A special LED for each direction illuminates when the viewing mirror is in the maximum position in this direction." \
                    "If a special LED is illuminated for each direction,when the exterior mirror is in the maximum position in this direction and the mirror is back, then When the exterior mirror heating is activated, the LED is lit." \
                    "If the exterior mirror heating is deactivated,the LED does not light up." \
                    "Deactivating the exterior mirror heating leads to the deactivation of the LED." \
                    "Pressing the button to close the window will cause the window to close as soon as the print is exerted or the window has reached its top." \
                    "Pressing the button to open the window will cause the window to open as long as the print is running or the window has reached the bottom." \
                    "Pressing the button to open the window will open the window completely." \
                    "Pressing the button to close the window will cause the window to close completely." \
                    "If the window is just opened, pressing the close button will cause the window to stop." \
                    "If the window is just opened, nothing happens by pressing the button." \
                    "If the window is closed, press the button to open the window." \
                    "If the window is closed, nothing happens by pressing the key." \
                    "When the window is completely open, pressing the button for opening does not have any effect." \
                    "If the window is completely closed, pressing the key for closing does not have any effect." \
                    "As soon as the window is closed and an obstacle in the window frame is detected by back pressure,the finger protection is activated." \
                    "If the anti-trap protection is activated, the window can no longer be closed." \
                    "The finger protection can be deactivated by briefly pressing the open button." \
                    "If the LED is present and the finger protection is activated, the LED is lit." \
                    "If the LED is present and the finger protection is not activated, the LED does not light up." \
                    "When the mirror is folded, pressing the central button will cause the mirror to fold out." \
                    "When the mirror is folded out and the vehicle is stationary,pressing the central button causes the mirror to fold." \
                    "When the mirror is unfolded and the vehicle is in motion,pressing the central button has no effect." \
                    "If the LED is present and the mirror is folded in, the LED is lit." \
                    "If the LED is present and the mirror is folded out, the LED does not light up." \
                    "Pressing the directional button left of the electrical exterior mirror,causes the exterior mirror to move to the left." \
                    "This only works when the exterior mirror has moving margin to the left." \
                    "Pressing the directional button right of the electrical exterior mirror causes the exterior mirror to move to the right." \
                    "This only works when the exterior mirror has movement margin to the right." \
                    "Pressing the directional button up of the electrical exterior mirror causes the exterior mirror to move upwards." \
                    "This only works when the exterior mirror has moving range up." \
                    "Pressing on the directional button down of the electrical exterior mirror causes the exterior mirror to move downwards. " \
                    "This only works when the exterior mirror has movement range down." \
                    "Two temperatures are defined,a minimum temperature and a maximum temperature." \
                    "If the temperature measurement on the exterior mirror falls below the minimum temperature,the exterior mirror heating is activated." \
                    "Switching off the vehicle leads to the switch-off of the exterior mirror heater." \
                    "When the exterior mirror heating is activated and the maximum temperature is exceeded,the exterior mirror heating is deactivated." \
                    "If the LED is present and the exterior mirror heating is activated, the LED is lit." \
                    "If the LED is present and the exterior mirror heater is not activated,the LED does not light up." \
                    "Violent opening of a door causes an alarm to be triggered when the alarm is active." \
                    "Violent opening of a door does not trigger an alarm when the alarm system is deactivated." \
                    "If an LED is present, an ongoing alarm is signaled by an LED." \
                    "Alarm is automatically terminated after 20 seconds,after which a message will be displayed if LEDs are present." \
                    "Alarm can be deactivated by unlocking the vehicle. " \
                    "This does not lead to a message when LEDs are present." \
                    "When recording a movement in the interior while the alarm system is activated, alarm is triggered." \
                    "The interior monitoring is deactivated as soon as the alarm system is deactivated." \
                    "If the LED for the alarm system is available, a special LED is lit when the interior monitoring is activated." \
                    "If the LED for the alarm system is present and the interior monitoring is deactivated,the special LED does not light up." \
                    "Pressing the locking button and inactive central locking activates the central locking system,which closes all doors." \
                    "Pressing the locking button and active central locking system will not trigger any action." \
                    "When pressing the unlocking button and active central locking system,the central locking system is deactivated, whereby all doors are unlocked." \
                    "No action is triggered when the unlocking button and the inactive central locking system are pressed." \
                    "By closing a door with inactive central locking, the central locking system is activated,whereby all doors are closed." \
                    "By closing a door with the central locking system active,no action is triggered." \
                    "By unlocking a door with active central locking system,the central locking system is deactivated, whereby all doors are unlocked." \
                    "By unlocking a door with inactive central locking system, no action is triggered." \
                    "When the window is locked, when the central locking system is inactive and the automatic power window is in operation, all windows are closed automatically,if they are open,otherwise they will be blocked." \
                    "When unlocking the window, active central locking system and the existence of the automatic power window, the blocking of all windows is canceled." \
                    "When the window is locked, if the central locking system is inactive and the manual power window is activated, all windows are blocked." \
                    "When unlocking the window, active central locking system and the existence of the manual power window, the blocking of all windows is canceled." \
                    "When a defined speed is exceeded, all doors are closed when they are not closed." \
                    "When a defined speed is exceeded, nothing happens if the doors have already been completed manually." \
                    "If the speed is reduced so that it falls below the defined value and the doors have been closed by the automatic locking feature, the doors are unlocked." \
                    "If the speed is reduced so that it falls below the defined value and the doors have not been manually closed by the automatic locking feature, the doors are not unlocked." \
                    "Pressing the remote control button to lock activates the central locking system if the central locking system is not activated." \
                    "Pressing the remote control button to unlock unlocks the central locking system when the central locking system is activated." \
                    "Pressing the remote control button for locking has no effect when the central locking is active." \
                    "Pressure on the remote control button for unlocking has no effect if the central locking is decanted." \
                    "By pressing the corresponding button on the remote control,the alarm system is activated when it was previously inactive." \
                    "If the alarm is active, pressing the remote control button does not have any effect." \
                    "By pressing the corresponding button on the remote control,the alarm system is deactivated if it was active before." \
                    "If the alarm is not active, pressing the remote control button does not have any effect." \
                    "The passage of ten seconds between the unlocking and the opening of a door leads to the door being closed again." \
                    "The unlocking and opening of a door within 10 seconds causes the door to not be automatically closed again." \
                    "Pressing the remote control button to open the window causes the window to open for opening." \
                    "Pressing the remote control button to close the window will allow the window to close up for closing." \
                    "If the window is being opened, pressing the remote control button will close the window." \
                    "If the window is just opened, nothing happens by pressing the button." \
                    "If the window is closed, press the remote control button to open the window." \
                    "If the window is closed, nothing happens by pressing the key." \
                    "If the window is fully open, pressing the remote control button will have no effect." \
                    "If the window is completely closed, the pressing of the remote control button for closing does not have any effect." \


x = ' '

text1 = "If the window is closed, nothing happens by pressing the key." \

word2vec = './word2vectrained'
# word2vec = './GoogleNews-vectors-negative300.bin'
# word2vec = None
extractor = KeywordExtractor(word2vec=word2vec)

keywords = extractor.extract(text, ratio=0.15, split=True,scores= True)
print(keywords)

global_keyword = []
sent_words = []
newkeys = []
newkeys1 = []
x = " "


for keyword in keywords:
    print(keyword[0])
    newkeys.append(keyword[0])

print(" ")

text1 = text.lower()
sentences = text1.split('.')

tree1 = []
sent1 = []
for keyword in keywords:
    global_keyword.append(keyword[0])
    tree1.append(keyword[0])
    # print(global_keyword)
    print(keyword)
    for sentence in sentences:
        if keyword[0] in sentence:
            # print(keyword[0])
            print(sentence)
            sent1.append(sentence)
            sent1.append('.')
            sentences.remove(sentence)
            # print(sentences)
    print(" ")


    Sub_Keyword = ''.join(sent1)
    # print(Sub_Keyword)
    Sentences1 = Sub_Keyword.split('.')
    Sub_Keyword2 = []
    extractor = KeywordExtractor(word2vec=word2vec)
    key1 = extractor.extract(Sub_Keyword, ratio=0.8, split=True, scores=True)
    # print(key1)
    for key in key1:
        global_keyword.append(key[0])
        newkeys1.append(key[0])
        if key[0] not in newkeys:
            print(" ")
            print("TREE Level 1 ")
            print("(The Sub-key Tree 1 words for this Extracted Keyword is:)")
            print(x,x,x,x,key[0])
            for sen in Sentences1:
                if key[0] in sen.lower():
                #   print("These are the sentences for:")
                #   print(key[0])
                    print(x,x,x,x,sen)
                    Sub_Keyword2.append(sen)
                    Sub_Keyword2.append('.')
                    Sentences1.remove(sen)
            Sub_Keyword3 = ''.join(Sub_Keyword2)
            Sentences2 = Sub_Keyword3.split('.')
            key2 = extractor.extract(Sub_Keyword3, ratio=0.8, split=True, scores=True)
            # print(key2)
            for k in key2:
                global_keyword.append(k[0])
                if k[0] not in newkeys1:
                    print(" ")
                    print(" ")
                    print(x, x, x, x, x, x, x, x, "TREE Level 2")
                    print(x, x, x, x, x, x, x, x, x, "(The Sub-key2 words for this Extracted Keyword is:) ")
                    print(x, x, x, x, x, x, x, x, x, k[0])
                    for sub in Sentences2:
                        if k[0] in sub.lower():
                            # print(k[0])
                            print(x, x, x, x, x, x, x, x, x,sub)
                            Sentences2.remove(sub)
            Sub_Keyword2 = []
            print(x)
    sent1 = []

print(" ")
print(" ")
print(len(set(global_keyword)))


abc = ['window','button','close','pressing','mirror','folded','cause','pressing']
S = ','.join(abc)
keywww = extractor.extract(S, ratio=1.0, split=False, scores=True)
print(keywww)



final_list=[]
for text in global_keyword:
    a=clustering.process_text(text, stem=True)
    for element in a:
        final_list.append(element)



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
          "Violent opening of a door causes an alarm to be triggered when the alarm is active.",
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

# clustering.articles = global_keyword
# clustering.process_text(final_list, stem=False)
print('===============First clustering algorithm===================')
kmeans = clustering.cluster_texts(format,clusters=20)
clustering.pprint(dict(kmeans))
print('===============First clustering algorithm===================')



print('===============second clustering algorithm===================')
# aggo = clustering.cluster_texts(format,clusters=12)
# clustering.pprint(dict(aggo))
print('===============second clustering algorithm===================')

# dbscan = clustering.cluster_texts(sentences,clusters=30)
# clustering.pprint(dict(dbscan))

# from jgtextrank import keywords_extraction
from gensim.summarization import keywords

clus = []
a=dict(kmeans)
for i,j in zip(a.values(),a.keys()):
    print('----------------cluster' + ' ' + str(j) + '-------------')
    for value in i:
        print(format[value])
        clus.append(format[value])
    print('-------------------cluster'+' '+ str(j) + '-------------')
    # print('Running textrank')
    # print('-------------------Textrank starting' + ' ' + str(j) + '-------------')
    # extractor = KeywordExtractor(word2vec= None)
    # clus_new = ''.join(clus)
    # stop_list = {'set', 'mixed', 'corresponding', 'supporting'}
    # custom_categories = {'NNS', 'NNP', 'NN'}
    # clus_key = keywords(clus_new,ratio=0.1, words=None, split=False, scores=False, pos_filter=('NN'),lemmatize=False, deacc=True).split('\n')
    # clus_key = extractor.extract(clus_new, ratio=0.50, split=True, scores=True)
    # print(clus_key)
    # print('-------------------Textrank End' + ' ' + str(j) + '-------------')



print('he')

# print(set(global_keyword))
# print(len(set(global_keyword)))











#-----------------------------------------------
    # Sub_Keyword3 = ''.join(Sub_Keyword2)
    # #print(Sub_Keyword3)
    # Sentences2 = Sub_Keyword3.split('.')
    # extractor = KeywordExtractor(word2vec=word2vec)
    # key1 = extractor.extract(Sub_Keyword3, ratio=0.8, split=True, scores=True)
    # for k in key1:
    #     x = ' '
    #     print(" ")
    #     print(" ")
    #     print(x, x, x, x, x, x, x, x,"TREE Level 2")
    #     print(x, x, x, x, x, x, x, x, x, "(The Sub-key2 words for this Extracted Keyword is:) ")
    #     print(x, x, x, x, x, x, x, x, x, k[0])
    #     for sub in Sentences2:
    #         if k[0] in sub.lower():
    #             # print(k[0])
    #             print(x, x, x, x, x, x, x, x, x,sub)
    #             Sentences2.remove(sub)
    #     Sub_Keyword2 = []
    #     print(" ")


#-----------------------------------------------

# sentences = text.split('.')
# # print(sentences)
# new_keyword = keywords
#
# for sentence in sentences:
#     for key in new_keyword:
#         if key in sentence:
#             print(new_keyword)
#             print(sentence)

#-----------------------------------------------

# sentences = text.split('.')
# # print(sentences)
# keywords = ['when','alarm']
#
# for sentence in sentences:
#     for keyword in keywords:
#         if keyword in sentence:
#             print(keyword)
#             print(sentence)


#-----------------------------------------------
# for sentence in sentences:
#     if keywords in sentence:
#         print(keywords)
#         print(sentence)

# for keywords in sentences:
#     if keywords in sentences:
#         print(sentences)
#         break
#
# out = re.search(r'between\s+(?P<champion>.*?)\s+\("champion"\)\s+and\s+(?P<underdog>.*?)\s+\("underdog"\)', text)
# #print(sentences)
#
# for k,i in sentences:
#     if k[i] == keywords:
#         print(k)

#-----------------------------------------------


