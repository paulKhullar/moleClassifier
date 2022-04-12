import os
import cv2
import pickle


#change as needed 
DIRECTORY_OF_DATA_FILE = ".\\data"
IMG_SIZE = 100 
CATEGORIES = ['benign','malignant']

#data wrangling is based on this:
#https://towardsdatascience.com/all-the-steps-to-build-your-first-image-classifier-with-code-cf244b015799

#this function assigns labels to either the test or the train set and resizes the image for our neural net
def create_train_test_dataset(test_or_train) :
    data = []
    for category in CATEGORIES :
        path = os.path.join(DIRECTORY_OF_DATA_FILE,test_or_train,category)
        #from now on, benign is 0 and malignant is 1 - this is a binary classification problem
        classification_number = CATEGORIES.index(category)
        for img in os.listdir(path) :
            try :
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)
                #resize our images to an arbitary constant, feel free to experiment with this number idk what will work best. Larger will take longer tho
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array,classification_number])
            except Exception as e:
                print("error")
    return data
###############################################################################################
#pickle is an object serializer library which dumps stuff in storage, storing our modified data means that we dont have to run the time consuming function above every time.
#we will do this for both the test and train data set.
#this could probably be done better but im lazy so fuck you

train_set_raw = create_train_test_dataset('train')
x_train = [] # training image
y_train = [] # training label

for image, label in train_set_raw :
    x_train.append(image)
    y_train.append(label)

pickle_out = open("x_train.pickle", "wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()


test_set_raw = create_train_test_dataset('test')
x_test = [] # test image
y_test = [] # test label

for image, label in test_set_raw :
    x_test.append(image)
    y_test.append(label)

pickle_out = open("x_test.pickle", "wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()
################################################################