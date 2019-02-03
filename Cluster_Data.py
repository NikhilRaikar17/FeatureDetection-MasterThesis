import warnings
import Clustering
import nltk
warnings.simplefilter(action='ignore', category=FutureWarning)
nltk.download('maxent_ne_chunker')
nltk.download('words')
import Feature_Generation


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import pyfpgrowth
import preprocessing

sample2 = ["The system displays all products which are available.",
 "The system allows customers to select the product to purchase.",
 "Customer selects his choice of the product by interacting with the system.",
 "Recommendations of products are given interms of per customer basis.",
 "Recommendations of combinations of products are given interms of per customer basis.",
 "By selecting the products a customer wishes to buy, he proceeds to the the payment option .",
 "Entering the shipping details will proceed the purchase order further.",
 "Collecting the shipping details the payment option is asked by the system.",
 "Different payment options are available.",
 "The system supports paying with credit cards.",
 "The system supports paying with gift cards.",
 "If a customer pays with a credit card, the system approves first the payment by contacting the credit card company.",
 "The payment option is verified and processed further."
 "When the system completes recording an order, the supplier can ship the ordered products to the customer.",
 "Order reciepts are sent to the customer via Email.",
 "Confirmation Emails of dispatch and delivery of the product to its destination are also sent via Email.",
 "Various Shipping options are also given to the customer.",
 "Customer registration option is available.",
 "If a customer resgisters and then buys a product,his account will be updated with the order details.",
 "Prices for various pruchases will vary accordingly for small and big products.",
 "Shipping costs are added for small products.",
 "Shipping costs are not added for large products.",
 "Air mail shipping allowed only for small products.",
 "If a customer pays with a gift card, the system supports only land shipping.",
 "Customer can track his purchase with his unique orderID.",
 "After the purchase is completed, the system will redirect the user to its main home page catalog.",
 "There is a review product option available to voice opinions of the products sold.",
 "As soon as a customer has purchased a product, a review product mail will be sent to the customer.",
 "Products are updated in the main page catalog by its relevant suppliers",]


sample = ["After the system verifies the purchase's payments details, the supplier confirms the purchase.",
            "The system asks for shipment details.",
	        "The system supports paying with credit cards.",
            "If a customer pays with a credit card, the system approves first the payment by contacting the credit card company.",
            "When the system completes recording an order, the supplier can ship the ordered products to the customer.",
            "The system sends the shipping documents via email.",
            "When the system finalizes a software order details, the supplier ships the ordered product via email.",
            "The system sends the shipping documents.",
            "The system supports different shipping options.",
            "However, if a customer buys a very small product, the system supports only air mail shipping.",
            "The system displays the available products.",
            "When a registered customer buys a product, the system updates the inventory.",
            "The system supports paying with gift cards.",
            "If a customer pays with a gift card, the system supports only land shipping.",
            "A customer can track the purchase status.",
            "The system provides details on the product delivery status.",
            "The system presents the product return page.",
            "If the customer returns a product, the system updates the inventory.",
            "The system presents the product page,When a supplier enters new products, the system updates the catalog.",
            "The system enables customers writing reviews on products.",
            "When a customer reviews a product, the system sends the product review to the relevant supplier.",
            "When a supplier receives new products, he enters the new products to the system.",
            "The system updates the catalog.",
            "When the system presents the available products list, a customer can purchase a product.",
            "The system updates the shopping cart.",
            "The system presents the ordering page.",]


# Dataset cleaned and PreProcessed
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

antivirus = ["""Antivirus Scanner:Protection Cloud Technology:PUA Shield:Award-winning protection from malware (viruses, Trojans, worms, etc.)Scans unknown files in real time for malware and exploits.'
             'Identifies potentially unwanted applicationsHidden programs bundled with other software.'
             'They display ads, slow down your PC and redirect you to other websites.hidden within legitimate software.'
             'Connected Home Monitor-Renamed from Home Network Protection and added a new feature to scan router-connected IoT devices (Internet of Things) to test for vulnerabilities such as weak passwords, open ports and known services.'
             'UEFI Scanner-Proactive scanning that runs in the background and only notifies you if a problem is detected.License Manager-New feature added to my.eset.com that allows you to view and manage your licenses and connected devices.'
             'License Manager-New feature added that allows you to view and manage your licenses and connected devices.'
             'License Manager-New feature added that allows you to view and manage your licenses and connected devices.'
             'Application upgrade in the background mode has been improved. '
             'You no longer have to accept the terms of the End User License Agreement again during the upgrade unless those terms have been changed. '
             'Mail Anti-Virus has been improved. '
             'The default level of heuristic analysis has been increased to mediumKaspersky Anti-Virus 2017 provides the following new features:Kaspersky Anti-Virus 2017 provides the following new features:Mail Anti-Virus has been improved. '
             'The default level of heuristic analysis has been increased to medium Threat-removal Layer - Targets and eliminates hard-to-remove threats less sophisticated products often miss.Always up-to-date Product Version - Norton automatically sends you important product and feature updates throughout the year. '
             'The latest version installs without you needing to do anything.FREE 24x7 Support - Offers you expert help and answers by phone, live chat or online, whenever you need them.1NEW:FREE 24x7 Support - Offers you expert help and answers by phone, live chat or online, whenever you need them. '
             'Performance improvements and fixes for the FG-VD-17-019 vulnerability reported by Fortinets FortiGuard LabsNEW:Performance improvements and fixes for the FG-VD-17-019 vulnerability reported by Fortinet's FortiGuard Labs30-day trial period Nag screenMain Program updated to 6.9.5.2956, FastScan to 6.9.5.1356.New: added detection for malware using Explorer.exe to automatically launch web pages.Fix: if a Scheduled Task was detected loading malware, the Scheduled Task was not being correctly deleted (although the loading malware would still have been disabled correctly).Helpfile updates.NEW:Helpfile updates.
             'Subscription and Device Management integrated in the product. Users can now manage their subscriptions (product management through My Account, Renewal and Upsell areas), as well as their anti-theft protection (My Devices area) from their products.' \
             'Fully compatibles with the NEW Windows 10 Anniversary Update.Engine improvements for better protection and performance.' \
             'ANTIVIRUS PRO,Engine improvements for better protection and performance.' \
             'Bug fixes:Basic protection only. ' \
             'Advanced features are available in AVG Internet Security and AVG Ultimate. ' \
             'Fixed Payment protection (Fake Website Shield component) , which was disabled after clean installation sometimes.' \
             'Bug fixes:Fixed Payment protection (Fake Website Shield component) , which was disabled after clean installation sometimes. ' \
             'Russian anti-virus company Doctor Web has updated Dr.Web Anti-rootkit API (11.1.5.201607180), Dr.Web Scanning Engine 11.1.2.201608020) and the virus databases in its Dr.Web CureIt! utility.' \
             'The update resolves an issue that caused the scanner to freeze if the utility was started using the option Run as.' \
             'Bug fixes,The update resolves an issue that caused the scanner to freeze if the utility was started using the option Run as. ' \
             'Entertainment mode has been introduced.' \
             'Using Entertainmentmode you will be able to play games, watch movies and use anyentertainment software without any interference by Quick Healand without reducing security of your computer.' \
             'Bug fixes:and without reducing security of your computer.' \
             'Several modules (Anti-Tracking, Camera Guard, USB Disk Guard, Malicious Action) are disabledAnti-ransomware engine cannot be activatedThe Bitdefender engine cannot be activatedAuto Scan is disabledAuto Update is disabled.' \
             'Optimized scan engine to repair the occasional scan errors for smoother scan.' \
             'Added optimization for User Account Control, Windows SmartScreen, and Windows Update under Security Reinforce.' \
             'Expanded database to remove the latest threats including Tabs 2 Grid, WinThruster, and Dolphin Deals.Improved multiple languages.' \
             'Bug fixes:Improved multiple languages. New and improved features:1000 machines limitNag screen Administrative rights on the local computerEnabled NetBIOS over TCP/IPEnabled File and Printer sharingEnabled access to the ADMIN$ shareAn ability to ping the computer within 1500 msStarted services: Computer Browser, Remote RegistryTCP ports opened: 135, 139, 445UDP ports opened: 137, 138The program user interface was changed to allow using UI skins. ' \
             'Now you can choose a preferred skin to change the way how the program looks.New and improved features:The program user interface was changed to allow using UI skins.' \
             ' Now you can choose a preferred skin to change the way how the program looks.30 days trial. ' \
             ' F-Secures Real-time Protection Network is an online service which provides rapid response against Internet-based threats.' \
             'The Real-time Protection Network uses reputation services to obtain information about the latest Internet threats. ' \
             ' You can help us develop the service further by contributing detailed information, such as sources of intrusive programs or messages, and behavioral and statistical analysis of the use of the computer and the Internet.' \
             'New and improved features:The Real-time Protection Network uses reputation services to obtain information about the latest Internet threats.' \
             'You can help us develop the service further by contributing detailed information, such as sources of intrusive programs or messages, and behavioral and statistical analysis of the use of the computer and the Internet."""]

#K-Means Clustering Algorithm has been Run on the PreProcessed data.
print('===============First clustering algorithm===================')
kmeans = Clustering.cluster_texts(format ,clusters=20)
Clustering.pprint(dict(kmeans))
print('===============First clustering algorithm===================')

kmd = Clustering.cluster_texts(format ,clusters=20)
Clustering.pprint(dict(kmd))



# spherical K-means
# spherical = Clustering.cluster_texts(format,clusters=20)
# Clustering.pprint(dict(spherical))


#Dbscan
# dbscan = Clustering.cluster_texts(format)
# Clustering.pprint(dict(dbscan))

# kmedoids = Clustering.cluster_texts(format,clusters=20)
# Clustering.pprint(dict(kmedoids))

# Affinity Propogation
# affinityprop = Clustering.cluster_texts(format)
# Clustering.pprint(dict(affinityprop))

# print('===============Second clustering algorithm===================')
# aggo = Clustering.cluster_texts(format,clusters=20)
# Clustering.pprint(dict(aggo))
# print('===============Second clustering algorithm===================')

#============================================================================================
#Named-Entity tags and POS-Tags has been made use of to derive the results.
from mlxtend.frequent_patterns import apriori
import itertools
#============================================================================================

clus = []
f = []
stopwords = ['.']
a=dict(kmd)
for i,j in zip(a.values(),a.keys()):
    print('----------------cluster' + ' ' + str(j) + '-------------')
    for value in i:
        print(format[value])
        clus.append(format[value])
        f.append(nltk.word_tokenize(format[value]))
    Feature_Generation.named_Entity(""" """.join(clus))
    clus = []
    print('-------------------cluster'+' '+ str(j) + '-------------')
    f = []








#============================================================================================
# Frequent pattern Mining algorithms
 # abc = list(itertools.chain.from_iterable(f))
    # print(f)
    # patterns = pyfpgrowth.find_frequent_patterns(f, 1)
    # print(patterns)
    # rules = pyfpgrowth.generate_association_rules(f, 0.7)
    # print(rules)
    # abc=list(itertools.chain.from_iterable(f))
    # te = TransactionEncoder()
    # te_ary = te.fit(f).transform(f)
    # df = pd.DataFrame(te_ary, columns=te.columns_)
    # frequent_itemsets = apriori(df, min_support=0.9, use_colnames=True)
    # frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    # print(frequent_itemsets)

#============================================================================================


# sentences = preprocessing.words_reviews
# print(sentences)
# te = TransactionEncoder()
# te_ary = te.fit(sentences).transform(sentences)
# df = pd.DataFrame(te_ary, columns=te.columns_)
# frequent_itemsets = apriori(df, min_support=0.8, use_colnames=True)
# frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
# print(frequent_itemsets)

#============================================================================================


