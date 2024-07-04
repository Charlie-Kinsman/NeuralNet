import cv2
#import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
cv2.namedWindow("Me")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    #gray_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2XYZ)
    #half = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    #half = frame[0:500,200:500]
    cv2.imshow("Me", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(0)
    if key == 32: # exit on ESC
        continue
    if key == 27:
        break

cv2.destroyWindow("Me")
vc.release()

class image():
    def __init__(self):
        vc = cv2.VideoCapture(0)
        if vc.isOpened():
            rval, frame = vc.read()
        self.frame = frame
        vc.release()
im = image()
print(im.frame)
"""
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print(type(train_images[0]))
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
"""