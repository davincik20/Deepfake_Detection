
# Refference link
# https://www.kaggle.com/competitions/deepfake-detection-challenge/data

from django.shortcuts import render,HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})

def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def UserTraining(request):
    train_frame_folder = 'train_sample_videos'
    # with open(os.path.join(train_frame_folder, 'metadata.json'), 'r') as file:
    #     data = json.load(file)
    # list_of_train_data = [f for f in os.listdir(train_frame_folder) if f.endswith('.mp4')]
    # detector = dlib.get_frontal_face_detector()
    # for vid in list_of_train_data:
    #     count = 0
    #     cap = cv2.VideoCapture(os.path.join(train_frame_folder, vid))
    #     frameRate = cap.get(5)
    #     while cap.isOpened():
    #         frameId = cap.get(1)
    #         ret, frame = cap.read()
    #         if ret != True:
    #             break
    #         if frameId % ((int(frameRate)+1)*1) == 0:
    #             face_rects, scores, idx = detector.run(frame, 0)
    #             for i, d in enumerate(face_rects):
    #                 x1 = d.left()
    #                 y1 = d.top()
    #                 x2 = d.right()
    #                 y2 = d.bottom()
    #                 crop_img = frame[y1:y2, x1:x2]
    #                 if data[vid]['label'] == 'REAL':
    #                     cv2.imwrite('media/dataset/real/'+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (128, 128)))
    #                 elif data[vid]['label'] == 'FAKE':
    #                     cv2.imwrite('media/dataset/fake/'+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (128, 128)))
    #                 count+=1
    import os
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    input_shape = (128, 128, 3)
    data_dir = 'media/dataset'

    real_data = [f for f in os.listdir(data_dir+'/real') if f.endswith('.png')]
    print('real data:')
    print(real_data)
    fake_data = [f for f in os.listdir(data_dir+'/fake') if f.endswith('.png')]
    print('fake data')
    print(fake_data)

    

    X = []
    Y = []

    for img in real_data:
        X.append(img_to_array(load_img(data_dir+'/real/'+img)).flatten() / 255.0)
        Y.append(1)

    print('for in real data')
    for img in fake_data:
        X.append(img_to_array(load_img(data_dir+'/fake/'+img)).flatten() / 255.0)
        Y.append(0)
    print('for in fake data')

    Y_val_org = Y

    #Normalization
    X = np.array(X)
    Y = to_categorical(Y, 2)
    print('done normalization')
    #Reshape
    X = X.reshape(-1, 128, 128, 3)

    #Train-Test split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)
    print('*'*100)
    print(X_train)

    from tensorflow.keras.applications import InceptionResNetV2
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    print('imported')


    googleNet_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    googleNet_model.trainable = True
    model = Sequential()
    model.add(googleNet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=2, activation='softmax'))
    print('layers added')
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.legacy.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                metrics=['accuracy'])
    print('compiled.....')
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=2,
                               verbose=0, mode='auto')
    EPOCHS = 20
    # EPOCHS = 2
    BATCH_SIZE = 100
    history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (X_val, Y_val), verbose = 1)
    acc = history.history['accuracy'][-1]
    loss = history.history['loss'][-1]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))
    t = f.suptitle('Pre-trained InceptionResNetV2 Transfer Learn with Fine-Tuning & Image Augmentation Performance ', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    epoch_list = list(range(1,EPOCHS+1))
    ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(0, EPOCHS+1, 1))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch #')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(0, EPOCHS+1, 1))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch #')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")

    return render(request,'users/training_result.html', {'acc':acc,'loss':loss})

def DetectDeepfake(request):
    return render(request, 'users/detect_deepfake.html')

def UploadImageAction(request):
    if request.method == 'POST':
        from django.core.files.storage import FileSystemStorage
        from keras.models import Model as KerasModel
        from keras.layers import Input, Dense,Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
        from keras.optimizers import Adam
        import shutil
        try:
            shutil.rmtree('media/test_images')
        except:
            pass
        # os.remove('media/test_images')
        myfile = request.FILES['file']
        fs = FileSystemStorage(location='media/test_images')
        filename = fs.save(myfile.name, myfile)
        print('file name:',filename)
        uploaded_file_url = fs.url(filename)
        # from .utility.deepfakeDetection import start_process
        # results = start_process(filename)
        messages.success(request, 'Image Processed Success')
        print("File Image Name " + uploaded_file_url)

        IMGWIDTH = 256
        print('modules imported')
        class Classifier:
            def __init__():
                self.model = 0
            
            def predict(self, x):
                return self.model.predict(x)
            
            def fit(self, x, y):
                return self.model.train_on_batch(x, y)
            
            def get_accuracy(self, x, y):
                return self.model.test_on_batch(x, y)
            
            def load(self, path):
                self.model.load_weights(path)

        class Meso4(Classifier):
            def __init__(self, learning_rate=0.001):
                self.model = self.init_model()
                optimizer = Adam(lr = learning_rate)
                self.model.compile(optimizer = optimizer, loss='mean_squared_error', metrics=['accuracy'])
            
            def init_model(self):
                x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))
                
                x1 = Conv2D(8, (3,3), padding='same', activation='relu')(x)
                x1 = BatchNormalization()(x1)
                x1 = MaxPooling2D(pool_size=(2,2), padding='same')(x1)
                
                x2 = Conv2D(8,(5,5), padding='same', activation='relu')(x1)
                x2 = BatchNormalization()(x2)
                x2 = MaxPooling2D(pool_size=(2,2), padding='same')(x2)
                
                x3 = Conv2D(16, (5,5), padding='same', activation='relu')(x2)
                x3 = BatchNormalization()(x3)
                x3 = MaxPooling2D(pool_size=(2,2), padding='same')(x3)
                
                x4 = Conv2D(16,(5,5), padding='same', activation='relu')(x3)
                x4 = BatchNormalization()(x4)
                x4 = MaxPooling2D(pool_size=(4,4), padding='same')(x4)
                
                y = Flatten()(x4)
                y = Dropout(0.5)(y)
                y = Dense(16)(y)
                y = LeakyReLU(alpha=0.1)(y)
                y = Dropout(0.5)(y)
                y = Dense(1, activation='sigmoid')(y)
                
                return KerasModel(x, y)
        print('built meso')
        from keras.preprocessing.image import ImageDataGenerator

        MesoNet_classifier = Meso4()
        MesoNet_classifier.load('media/Meso4_DF')

        print('model loaded')

        image_data_generator = ImageDataGenerator(rescale=1.0/255)
        data_generator = image_data_generator.flow_from_directory("media/", classes=["test_images"])
        print('images loaded')
        num_to_label = {1:"real", 0:"fake"}

        X, y = data_generator.next()
        probabilistic_predictions = MesoNet_classifier.predict(X)
        predictions = [num_to_label[round(x[0])] for x in probabilistic_predictions]
        print(predictions)
        # return HttpResponse(predictions)

        # return render(request, "users/uploadapicform.html", {"caption": results,'path':uploaded_file_url})
        return render(request, "users/detect_deepfake.html", {'prediction':predictions})


    