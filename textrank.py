from jgtextrank import preprocessing, build_cooccurrence_graph
from jgtextrank import keywords_extraction
import networkx as nx
import nltk
#from nltk import word_tokenize, pos_tag, ne_chunk

cluster6 = """Deactivation of the finger protection causes the LED to be deactivated.
            The finger protection can be deactivated by briefly pressing the open button.
            If the LED is present and the finger protection is activated, the LED is lit.
            If the LED is present and the finger protection is not activated, the LED does not light up.
            If the LED is present and the exterior mirror heater is not activated,the LED does not light up.
            This does not lead to a message when LEDs are present."""

cluster1 = """When the central locking system is active and the vehicle is closed, the LED lights up.
            If the central locking system is inactive and the vehicle is not closed,the LED does not light up.
            Deactivating the central locking system also turns off the LED.
            Alarm can be deactivated by unlocking the vehicle. 
            Pressing the locking button and inactive central locking activates the central locking system,which closes all doors.
            When pressing the unlocking button and active central locking system,the central locking system is deactivated, whereby all doors are unlocked.
            By closing a door with inactive central locking, the central locking system is activated,whereby all doors are closed.
            By unlocking a door with active central locking system,the central locking system is deactivated, whereby all doors are unlocked.
            Pressing the remote control button to lock activates the central locking system if the central locking system is not activated.
            Pressing the remote control button to unlock unlocks the central locking system when the central locking system is activated."""

cluster4 = """When all windows are still, the LED does not light.
            The LED is lit when the exterior mirror is folded.
            The LED does not light up when the exterior mirror is folded out.
            When the mirror is folded, pressing the central button will cause the mirror to fold out.
            When the mirror is folded out and the vehicle is stationary,pressing the central button causes the mirror to fold.
            If the LED is present and the mirror is folded in, the LED is lit.
            If the LED is present and the mirror is folded out, the LED does not light up."""

cluster7 = """When the button for closing a window is pressed and the corresponding window moves up,the LED indicates that a window closes.
            Closing of the opening or closing process of the windows leads to deactivation of the LED.
            Pressing the button to close the window will cause the window to close as soon as the print is exerted or the window has reached its top.
            Pressing the button to close the window will cause the window to close completely.
            Pressing the remote control button to close the window will allow the window to close up for closing."""

cluster0 = """When the button for opening a window is pressed and the corresponding window moves down,the LED indicates that a window is opening.
            Pressing the button to open the window will cause the window to open as long as the print is running or the window has reached the bottom.
            Pressing the button to open the window will open the window completely.
            If the window is just opened, pressing the close button will cause the window to stop.
            If the window is closed, press the button to open the window.
            When the window is completely open, pressing the button for opening does not have any effect.
            Pressing the remote control button to open the window causes the window to open for opening.
            If the window is being opened, pressing the remote control button will close the window.
            If the window is closed, press the remote control button to open the window.
            If the window is fully open, pressing the remote control button will have no effect."""

cluster12 = """Switching between the states of the viewing mirror leads to the change between the states of the LED.
Switching off the vehicle leads to the switch-off of the exterior mirror heater."""

cluster3 = """A special LED for each direction illuminates when the viewing mirror is in the maximum position in this direction.
            Pressing the directional button left of the electrical exterior mirror,causes the exterior mirror to move to the left.
            Pressing the directional button right of the electrical exterior mirror causes the exterior mirror to move to the right.
            Pressing the directional button up of the electrical exterior mirror causes the exterior mirror to move upwards.
            Pressing on the directional button down of the electrical exterior mirror causes the exterior mirror to move downwards."""

cluster15 = """If a special LED is illuminated for each direction,when the exterior mirror is in the maximum position in this direction and the mirror is back, then When the exterior mirror heating is activated, the LED is lit.
If the exterior mirror heating is deactivated,the LED does not light up.
Deactivating the exterior mirror heating leads to the deactivation of the LED.
If the temperature measurement on the exterior mirror falls below the minimum temperature,the exterior mirror heating is activated.
When the exterior mirror heating is activated and the maximum temperature is exceeded,the exterior mirror heating is deactivated.
If the LED is present and the exterior mirror heating is activated, the LED is lit."""

cluster5 = """If the window is just opened, nothing happens by pressing the button.
If the window is closed, nothing happens by pressing the key.
If the window is just opened, nothing happens by pressing the button.
If the window is closed, nothing happens by pressing the key."""

cluster11 = """If the window is completely closed, pressing the key for closing does not have any effect.
Pressing the remote control button for locking has no effect when the central locking is active.
Pressure on the remote control button for unlocking has no effect if the central locking is decanted.
If the alarm is active, pressing the remote control button does not have any effect.
If the alarm is not active, pressing the remote control button does not have any effect.
If the window is completely closed, the pressing of the remote control button for closing does not have any effect."""

cluster16 = """As soon as the window is closed and an obstacle in the window frame is detected by back pressure,the finger protection is activated.
If the anti-trap protection is activated, the window can no longer be closed."""


cluster18 = """When the mirror is unfolded and the vehicle is in motion,pressing the central button has no effect."""

cluster14 = """Two temperatures are defined,a minimum temperature and a maximum temperature.
When a defined speed is exceeded, all doors are closed when they are not closed.
When a defined speed is exceeded, nothing happens if the doors have already been completed manually.
If the speed is reduced so that it falls below the defined value and the doors have been closed by the automatic locking feature, the doors are unlocked.
If the speed is reduced so that it falls below the defined value and the doors have not been manually closed by the automatic locking feature, the doors are not unlocked."""


cluster13 = """Violent opening of a door causes an alarm to be triggered when the alarm is active.
Violent opening of a door does not trigger an alarm when the alarm system is deactivated.
When recording a movement in the interior while the alarm system is activated, alarm is triggered."""

cluster2 = """If an LED is present, an ongoing alarm is signaled by an LED.
Alarm is automatically terminated after 20 seconds,after which a message will be displayed if LEDs are present.
The interior monitoring is deactivated as soon as the alarm system is deactivated.
If the LED for the alarm system is available, a special LED is lit when the interior monitoring is activated.
If the LED for the alarm system is present and the interior monitoring is deactivated,the special LED does not light up."""

cluster9 = """Pressing the locking button and active central locking system will not trigger any action.
No action is triggered when the unlocking button and the inactive central locking system are pressed.
By closing a door with the central locking system active,no action is triggered.
By unlocking a door with inactive central locking system, no action is triggered."""

cluster10 = """When the window is locked, when the central locking system is inactive and the automatic power window is in operation, all windows are closed automatically,if they are open,otherwise they will be blocked.
When unlocking the window, active central locking system and the existence of the automatic power window, the blocking of all windows is canceled.
When the window is locked, if the central locking system is inactive and the manual power window is activated, all windows are blocked.
When unlocking the window, active central locking system and the existence of the manual power window, the blocking of all windows is canceled."""


cluster8 = """This only works when the exterior mirror has moving margin to the left.
This only works when the exterior mirror has movement margin to the right.
This only works when the exterior mirror has moving range up.
This only works when the exterior mirror has movement range down."""


cluster19 = """By pressing the corresponding button on the remote control,the alarm system is activated when it was previously inactive.
By pressing the corresponding button on the remote control,the alarm system is deactivated if it was active before."""

cluster17 = """The passage of ten seconds between the unlocking and the opening of a door leads to the door being closed again.
The unlocking and opening of a door within 10 seconds causes the door to not be automatically closed again."""

if __name__ == '__main__':
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    text = nltk.word_tokenize(cluster13)
    q = nltk.ne_chunk(nltk.pos_tag(text))
    # q.draw()
    # print(q)
    from nltk.chunk import conlltags2tree, tree2conlltags
    iob_tagged = tree2conlltags(q)
    # print(iob_tagged[0])
    w1 = []
    w2 = []
    noun_add = []
    for j in range(0, len(iob_tagged)-1):
        if(iob_tagged[j][1] == 'NNP'):
            noun = iob_tagged[j]
            # print(noun)
            w2 .append(noun)
            # print(w2)

    for i in range(0, len(iob_tagged) - 1):
        if (iob_tagged[i][1] == 'JJ' and iob_tagged[i + 1][1] == 'NN'):
            w = iob_tagged[i] + iob_tagged[i+1]
            w1.append(w)

            # print(iob_tagged[i] + iob_tagged[i+1])
        if(iob_tagged[i][1] == 'NN' and iob_tagged[i + 1][1] == 'NN'):
            noun = iob_tagged[i] + iob_tagged[i + 1]
            noun_add.append(noun)


    fre = nltk.FreqDist(w2)
    for word, frequency in fre.most_common(5):
        print(u'{};{}'.format(word, frequency))

    fre = nltk.FreqDist(noun_add)
    for word, frequency in fre.most_common(5):
        print(u'{};{}'.format(word, frequency))

    fre = nltk.FreqDist(w1)
    for word, frequency in fre.most_common(5):
        print(u'{};{}'.format(word, frequency))

    fre = nltk.FreqDist(w2)
    for word, frequency in fre.most_common(15):
        print(u'{};{}'.format(word, frequency))

    # ne_tree = conlltags2tree(iob_tagged)
    # print(ne_tree)


    preprocessed_context = preprocessing(cluster4, lemma=True)
    stop_list = {'set', 'system', 'temperature', 'button'}
    custom_categories = {'NN'}
    abc = keywords_extraction(cluster4, window=150,top_p =1, top_t=None,directed=False,stop_words=stop_list,syntactic_categories=custom_categories, lemma=True)[0][:30]
    cooccurrence_graph, context_tokens = build_cooccurrence_graph(preprocessed_context, window=150)
    # pos = nx.spring_layout(cooccurrence_graph,k=0.20,iterations=30)
    print(abc)




