from tensorflow import keras
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


name = 'mybin_2L'
mat_contents1 = sio.loadmat('XTrain_' + name + '.mat')
mat_contents2 = sio.loadmat('YTest_' + name + '.mat')
mat_contents3 = sio.loadmat('YTrain_' + name + '.mat')
mat_contents4 = sio.loadmat('XTest_' + name + '.mat')

XTrain_temp = mat_contents1['XTrain']
XTest_temp = mat_contents4['XTest']
YTest_temp = mat_contents2['YTest']
YTrain_temp = mat_contents3['YTrain']

imageSize = np.array(np.shape(XTrain_temp)[0:4])
imageSize_test = np.array(np.shape(XTest_temp)[0:4])
upscaledimageSize = np.array(np.shape(YTrain_temp)[0:3])

numTrainImages = imageSize[-1]
numImages_test = imageSize_test[-1]

XTrain = np.zeros((numTrainImages, imageSize[0], imageSize[1], 2))
for i in range(numTrainImages):
    XTrain[i, :, :, :] = XTrain_temp[:, :, :, i]

XTest = np.zeros((numImages_test, imageSize[0], imageSize[1], 2))
for i in range(numImages_test):
    XTest[i, :, :, :] = np.reshape(XTest_temp[:, :, :, i], (1, imageSize[0], imageSize[1], 2))

YTrain = np.zeros((numTrainImages, np.shape(YTest_temp)[0],np.shape(YTest_temp)[1], 3))
for i in range(numTrainImages):
    for j in range(3):
        YTrain[i, :, :, j] = YTrain_temp[:, :, j, i]

YTest = np.zeros((numImages_test, np.shape(YTest_temp)[0],np.shape(YTest_temp)[1], 3))
for i in range(numImages_test):
    for j in range(3):
        YTest[i, :, :, j] = YTest_temp[:, :, j, i]

# ------------layers
input_img = keras.Input(shape=(np.shape(XTrain)[1:]))
x = keras.layers.Conv2D(200, (int(imageSize[0]/upscaledimageSize[0]), int(imageSize[1]/upscaledimageSize[1])), padding='same', activation='relu', strides=(int(imageSize[0]/upscaledimageSize[0]), int(imageSize[1]/upscaledimageSize[1])))(input_img)
out_img = keras.layers.Conv2D(3, (1, 1), padding='same')(x)

# ---------------------------------
# model based on layers
ups = keras.Model(input_img, out_img)
ups.compile(optimizer='adam', loss='MeanSquaredError')
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# training
history = ups.fit(XTrain, YTrain,
                epochs=30,
                batch_size=int(0.1*numTrainImages),
                shuffle=True,
                validation_data=(XTest, YTest),
                callbacks=None,
                )
ups.save('model')

print("Evaluate on test data")
results = ups.evaluate(XTest, YTest)
print("test loss, test acc:", results)


mdict = {"Y_rec_test": ups.predict(XTest), "Y_rec_train": ups.predict(XTrain)}
sio.savemat("Y_rec" + name + ".mat", mdict)

sio.savemat('History.mat', {"history": history.history})

plt.figure(figsize=(20, 4))
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'],'--')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_results_semilogy_' + name + '.png')


for j in range(numImages_test):
    n = 3
    plt.figure(figsize=(20, 8))
    for i in range(2):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(XTest[j, :, :, i], norm=matplotlib.colors.Normalize())
        plt.title("Input")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for i in range(n):
        ax = plt.subplot(3, n, i + 4)
        plt.imshow(YTest[j, :, :, i], norm=matplotlib.colors.Normalize())
        plt.title("original_upscaled")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(3, n, i + 4 + n)
        plt.imshow(ups.predict(XTest[j:j+1, :, :, :])[0,:,:,i], norm=matplotlib.colors.Normalize())
        plt.title("reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig('results_test' + str(j) + '.png')

