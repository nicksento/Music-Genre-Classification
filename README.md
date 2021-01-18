# Music Genre Classification

In this project we use a dataset of audio files to build an RNN-LSTM network for Music Genre Classification.

The dataset that we are going to use is the GTZAN Dataset which can be found on [Kaggle](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification).

The GitHub repository of this project can be found [here](https://github.com/nicksento/Music-Genre-Classification).

# Contents
[1.Preprocessing Audio Data for Deep Learning](#prepro)      
[2.Implement an RNN-LSTM network for Music Genre Classification](#rnn)      
[3.Tune the Hyperparameters](#tune)          
[4.Conclusion, Future Work and Acknowledgements](#conc)         
[5.References](#ref)

# Preprocessing Audio Data for Deep Learning <a id='prepro'></a>

The GTZAN dataset is the most-used public dataset for evaluation in machine listening research for music genre recognition (MGR). The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions [1](#1).

The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

One way to classify data is through neural networks. Because NNs usually take in some sort of image representation, we will extract the Mel-frequency cepstral coefficients (MFCCs) from the audio files .

In sound processing, the mel-frequency cepstrum (MFC) is a representation of the shortterm power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.

Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC. They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum"). The difference between the cepstrum and the mel-frequency cepstrum is that in the MFC, the frequency bands are equally spaced on the mel scale, which approximates the human auditory system's response more closely than the linearly-spaced frequency bands used in the normal cepstrum. This frequency warping can allow for better representation of sound, for example, in audio compression [2](#2), [3](#3). 

MFCCs are commonly derived as follows:

1. Take the Fourier transform of (a windowed excerpt of) a signal.
2. Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows.
3. Take the logs of the powers at each of the mel frequencies.
4. Take the discrete cosine transform of the list of mel log powers, as if it were a signal.
5. The MFCCs are the amplitudes of the resulting spectrum.

```python
def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along with genre
        labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                try:
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                     
                    # process all segments of audio file
                    for d in range(num_segments):

                        # calculate start and finish sample for current segment
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        # extract mfcc
                        mfcc = librosa.feature.mfcc(signal[start:finish],
                                                    sample_rate,
                                                    n_mfcc=num_mfcc,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length)
                        mfcc = mfcc.T

                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            #print("{}, segment:{}".format(file_path, d+1))
                
                except:
                    print(file_path, 'error')
                    pass
                        
    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        print('Finished')
```

# Implement an RNN-LSTM network for Music Genre Classification <a id='rnn'></a>

A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs [4](#4), [5](#5), [6](#6).

Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). 

A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.

LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. LSTMs were developed to deal with the vanishing gradient problem that can be encountered when training traditional RNNs. Relative insensitivity to gap length is an advantage of LSTM over RNNs, hidden Markov models and other sequence learning methods in numerous applications [7](#7), [8](#8), [9](#9).

```python
def build_model(input_shape):
    """Generates RNN-LSTM model"""
    
    # create model
    model = keras.Sequential()
    
    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, 
                                input_shape=input_shape, 
                                return_sequences=True))
    model.add(keras.layers.LSTM(64))
    
    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    
    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model
```



```python
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, 
             loss="sparse_categorical_crossentropy",
             metrics=['accuracy'])
model.summary()
```

![image.png](attachment:image.png)

### Plot accuracy and error evaluation

![image-2.png](attachment:image-2.png)

```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
```
79/79 - 2s - loss: 1.0702 - accuracy: 0.6239

Test accuracy: 0.6239487528800964

The test accuracy is 62.4% which is not very high. We will try to tune the hyperparameters to improve our model.

# Tune the Hyperparameters <a id='tune'></a>

In machine learning, hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a parameter whose value is used to control the learning process. By contrast, the values of other parameters (typically node weights) are learned.

The same kind of machine learning model can require different constraints, weights or learning rates to generalize different data patterns. These measures are called hyperparameters, and have to be tuned so that the model can optimally solve the machine learning problem. Hyperparameter optimization finds a tuple of hyperparameters that yields an optimal model which minimizes a predefined loss function on given independent data. The objective function takes a tuple of hyperparameters and returns the associated loss. Cross-validation is often used to estimate this generalization performance [10](#10), [11](#11).

To tune the hyperparameters we are going to use *Keras Tuner* and *Bayesian Optimization*.

### Keras Tuner

The *Keras Tuner* is a library that helps you pick the optimal set of hyperparameters for your TensorFlow program. The process of selecting the right set of hyperparameters for your machine learning (ML) application is called hyperparameter tuning or hypertuning.

Hyperparameters are the variables that govern the training process and the topology of an ML model. These variables remain constant over the training process and directly impact the performance of your ML program. Hyperparameters are of two types [12](#12):

1.Model hyperparameters which influence model selection such as the number and width of hidden layers    
2.Algorithm hyperparameters which influence the speed and quality of the learning algorithm such as the learning rate for Stochastic Gradient Descent (SGD) and the number of nearest neighbors for a k Nearest Neighbors (KNN) classifier

### Bayesian optimization

*Bayesian optimization* is a global optimization method for noisy black-box functions. Applied to hyperparameter optimization, Bayesian optimization builds a probabilistic model of the function mapping from hyperparameter values to the objective evaluated on a validation set. By iteratively evaluating a promising hyperparameter configuration based on the current model, and then updating it, Bayesian optimization, aims to gather observations revealing as much information as possible about this function and, in particular, the location of the optimum. It tries to balance exploration (hyperparameters for which the outcome is most uncertain) and exploitation (hyperparameters expected close to the optimum). In practice, Bayesian optimization has been shown to obtain better results in fewer evaluations compared to grid search and random search, due to the ability to reason about the quality of experiments before they are run [13](#13), [14](#14). 

```python
def build_model(hp):
    """Generates RNN-LSTM model"""
    
    # create model
    model = keras.Sequential()
    
    # 2 LSTM layers
    model.add(keras.layers.LSTM(units=hp.Int('units',min_value=32,max_value=512,step=32),
                                input_shape=input_shape, 
                                return_sequences=True))
    model.add(keras.layers.LSTM(units=hp.Int('units',min_value=32,max_value=512,step=32)))
    
    # dense layer
    model.add(keras.layers.Dense(units=hp.Int('units',min_value=32,max_value=512), activation='relu'))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout', 0, 0.5, step=0.1, default=0.3)))
    
    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
              
    
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])), 
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    
    return model
    
bayesian_opt_tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=1,
    executions_per_trial=1,
    directory = LOG_DIR,
    project_name='kerastuner_bayesian_poc',
    overwrite=True)

bayesian_opt_tuner.search(X_train, y_train,epochs=50,
                          batch_size=1,
     validation_data=(X_validation, y_validation),
     #validation_split=0.2,
                    verbose=1)
```
![image-3.png](attachment:image-3.png)

![image.png](attachment:image.png)

The **validation accuracy** is **86.38%**.

```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
```
![image.png](attachment:image.png)

The **test accuracy** is **84.9%**. Hyperparameter tuning lead to more than 20% improvement on the test accuracy which is quite significant.

# Conclusion, Future Work and Acknowledgements  <a id='conc'></a>

We took a dataset consisting of audio files in .wav format and extracted the Mel-frequency cepstral coefficients (MFCCs).

We built a Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN), trained it using the MFCCs we extracted, and evaluated using a cross-validation and a test set.

We tuned its hyperparameters using Keras Tuner and Bayesian Optimization to obtain a better performance and re-evaluated the model resulting in **84.9%** test accuracy. 

### Further Work

The next step of this project is to deploy the model to cloud using Flask and Docker.

### Acknowledgements

Part of the code is from Valerio Velardo's [Youtube Channel](https://www.youtube.com/c/ValerioVelardoTheSoundofAI/featured) and [GitHub](https://github.com/musikalkemist).

# References  <a id='ref'></a>

[1]<a id='1'></a> [http://marsyas.info/downloads/datasets.html](http://marsyas.info/downloads/datasets.html)    

[2]<a id='2'></a> Min Xu; et al. (2004). ["HMM-based audio keyword generation"](https://web.archive.org/web/20070510193153/http://cemnet.ntu.edu.sg/home/asltchia/publication/AudioAnalysisUnderstanding/Conference/HMM-Based%20Audio%20Keyword%20Generation.pdf) (PDF). In Kiyoharu Aizawa; Yuichi Nakamura; Shin'ichi Satoh (eds.). Advances in Multimedia Information Processing – PCM 2004: 5th Pacific Rim Conference on Multimedia. Springer. ISBN 978-3-540-23985-7.

[3]<a id='3'></a> Sahidullah, Md.; Saha, Goutam (May 2012). "Design, analysis and experimental evaluation of block based transformation in MFCC computation for speaker recognition". Speech Communication. 54 (4): 543–565. [doi:10.1016/j.specom.2011.11.004](https://www.sciencedirect.com/science/article/abs/pii/S0167639311001622?via%3Dihub).

[4] <a id='4'></a> Dupond, Samuel (2019). "A thorough review on the current advance of neural network structures". Annual Reviews in Control. 14: 200–230.

[5] <a id='5'></a>Abiodun, Oludare Isaac; Jantan, Aman; Omolara, Abiodun Esther; Dada, Kemi Victoria; Mohamed, Nachaat Abdelatif; Arshad, Humaira (2018-11-01). ["State-of-the-art in artificial neural network applications: A survey"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6260436/). Heliyon. 4 (11): e00938. doi:10.1016/j.heliyon.2018.e00938. ISSN 2405-8440. PMC 6260436. PMID 30519653.

[6] <a id='6'></a>Tealab, Ahmed (2018-12-01). ["Time series forecasting using artificial neural networks methodologies: A systematic review"](https://www.sciencedirect.com/science/article/pii/S2314728817300715). Future Computing and Informatics Journal. 3 (2): 334–340. doi:10.1016/j.fcij.2018.10.003. [ISSN 2314-7288](https://www.worldcat.org/title/future-computing-and-informatics-journal-fcij/oclc/973838379).

[7] <a id='7'></a>Sepp Hochreiter; Jürgen Schmidhuber (1997). ["Long short-term memory"](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory). Neural Computation. 9 (8): 1735–1780. doi:10.1162/neco.1997.9.8.1735. PMID 9377276. S2CID 1915014.

[8] <a id='8'></a>Graves, A.; Liwicki, M.; Fernandez, S.; Bertolami, R.; Bunke, H.; Schmidhuber, J. (2009). ["A Novel Connectionist System for Improved Unconstrained Handwriting Recognition"](http://people.idsia.ch/~juergen/tpami_2008.pdf) (PDF). IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (5): 855–868. CiteSeerX 10.1.1.139.4502. doi:10.1109/tpami.2008.137. PMID 19299860. S2CID 14635907.

[9] <a id='9'></a>Sak, Hasim; Senior, Andrew; Beaufays, Francoise (2014). ["Long Short-Term Memory recurrent neural network architectures for large scale acoustic modeling"](https://web.archive.org/web/20180424203806/https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf) (PDF). Archived from the original (PDF) on 2018-04-24.

[10] <a id='10'></a>Claesen, Marc; Bart De Moor (2015). ["Hyperparameter Search in Machine Learning"](https://arxiv.org/abs/1502.02127). arXiv:1502.02127 [cs.LG].


[11] <a id='11'></a>Bergstra, James; Bengio, Yoshua (2012). ["Random Search for Hyper-Parameter Optimization"](https://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf) (PDF). Journal of Machine Learning Research. 13: 281–305.

[12] <a id='11'></a>[https://www.tensorflow.org/tutorials/keras/keras_tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner)

[13] <a id='11'></a> Hutter, Frank; Hoos, Holger; Leyton-Brown, Kevin (2011), ["Sequential model-based optimization for general algorithm configuration"](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/papers/11-LION5-SMAC.pdf) (PDF), Learning and Intelligent Optimization, Lecture Notes in Computer Science, 6683: 507–523, CiteSeerX 10.1.1.307.8813, doi:10.1007/978-3-642-25566-3_40, ISBN 978-3-642-25565-6

[14] <a id='11'></a>Bergstra, James; Bardenet, Remi; Bengio, Yoshua; Kegl, Balazs (2011), ["Algorithms for hyper-parameter optimization"](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf) (PDF), Advances in Neural Information Processing Systems
