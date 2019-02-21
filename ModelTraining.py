import os
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import backend as K


def modelTraing():

    base_model=MobileNet(weights='imagenet',include_top=False)
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x)
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    #x=Dense(64,activation='relu')(x) #dense layer 4
    preds=Dense(5,activation='softmax')(x)

    model=Model(inputs=base_model.input,outputs=preds)

    for layer in model.layers[:20]:
        layer.trainable=True

    for layer in model.layers[20:]:
        layer.trainable=True

    train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True)
    train_generator=train_datagen.flow_from_directory(r'./data/car/train/Audi/',
                                                      target_size=(224,224),
                                                      color_mode='rgb',
                                                      batch_size=20,
                                                      class_mode='categorical',
                                                      shuffle=True)

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    step_size_train=train_generator.n//train_generator.batch_size
    model.fit_generator(generator=train_generator,
                       steps_per_epoch=step_size_train,
                       epochs=1)

    return model

from keras.models import load_model
def readModelFile():
    model=load_model('my_model.hdf5')
    return model

def accuracyTest(model):
    classi = 0
    totalimg = 0
    correctpred = 0
    ary = {}

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    for dirname in os.listdir("./data/car/train/Audi"):
        print(dirname)
        test_generator = test_datagen.flow_from_directory(r"./data/car/test/car/",
                                                          # this is where you specify the path to the main data folder
                                                          target_size=(224, 224),
                                                          color_mode='rgb')

        y_prob = model.predict_generator(generator=test_generator, steps=1)
        pclasses = y_prob.argmax(axis=-1)
        totalimg = totalimg + len(pclasses)
        correctpred = correctpred + sum((y_prob.argmax(axis=-1) - classi) == 0)
        classi = classi + 1
        dim = dirname
        print("Test Accuracy " + dim + " = %.2f Percent" % (100.*correctpred/totalimg))
        ary[dim] = (100. * correctpred / totalimg)

    a1 = ary
    a1_sorted_keys = sorted(a1, key=a1.get, reverse=True)

    result=''
    x = 0
    for r in a1_sorted_keys:
        if x == 0:
            #print("Tested Car is : " + r + " With Accuracy : ", a1[r], "%")
            result="Tested Car is : " + r + " With Accuracy : ", a1[r], "%"
            carBrand=r
            carAccuracy=a1[r]
        if x == 1:
            print('')
            # print("Tested Car Second Probability is : " + r + " With Accuracy : ", a1[r])
        x = x + 1
    K.clear_session()
    return carBrand,carAccuracy


#------------------step 2---------------------------
# Find image type if side / Front / Rear

def CarViewChecking(brandName):
    #print("inside function")
    print(brandName)
    base_model = MobileNet(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    preds=Dense(3,activation='softmax')(x)
    #model = Sequential()
    model=Model(inputs=base_model.input,outputs=preds)
    for layer in model.layers[:20]:
        layer.trainable=True
    for layer in model.layers[20:]:
        layer.trainable=True
    train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True)
    train_generator=train_datagen.flow_from_directory(r'./data/car/train/View/'+brandName+'/',
                                                       target_size=(224,224),
                                                       color_mode='rgb',
                                                       batch_size=20,
                                                       class_mode='categorical',
                                                       shuffle=True)
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    step_size_train=train_generator.n//train_generator.batch_size
    if (step_size_train == 0):
        step_size_train = 3
    model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train,epochs=1)
    return train_generator,model


#-----------------------step 2 End -----------


def checkTestModel(train_generator,model):
        classi = 0
        totalimg = 0
        correctpred = 0
        ary={}
        test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
        toArry=train_generator.class_indices
        sortAr=sorted(toArry.items())
        for dirname in sortAr:
            test_generator=test_datagen.flow_from_directory("./data/car/test/car/", # this is where you specify the path to the main data folder
                                                             target_size=(224,224),
                                                             color_mode='rgb')
            y_prob = model.predict_generator(generator=test_generator, steps = 1)
            pclasses = y_prob.argmax(axis=-1)
            totalimg = totalimg + len(pclasses)
            correctpred = correctpred + sum((y_prob.argmax(axis=-1) - classi)==0)
            classi = classi + 1
            dim=dirname[0]
            ary[dim] = (100. * correctpred / totalimg)

        a1=ary
        a1_sorted_keys = sorted(a1, key=a1.get, reverse=True)
        brandName=''
        accuracy=""
        x=0
        for r in a1_sorted_keys:
            if x==0:
                #print("Tested Car is : " + r + " With Accuracy : ", a1[r] , "%")
                brandName=r
                accuracy=str(a1[r])
            x=x+1
        K.clear_session()
        return brandName,accuracy;

from keras.models import load_model


#mo=modelTraing()
#a,b=accuracyTest(mo)
#print(a)
#c,d=CarViewChecking(a)
#print(c)
#e,f=checkTestModel(c,d)

#print(e)