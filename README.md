# Dolphin
     
Dolphin: Deep neural networks on Intel multicore and MIC architecture ( Intel Xeon Phi Coprocessor).

     
## Brief Introduction
     
Currently, we have implemented the parallel **Stacked Autoencoder** and **Restricted Boltzmann Machine (RBM)** training algorithm on Intel Xeon & Xeon Phi platforms. Besides that, to better demo the usage of the components, we also combined the Stacked Autoencoders** with a **Softmax** classifier neural network.
     
Basically, the program loads both the training dataset and the testing dataset, and trains a neural network by stacked autoencoder algorithm. We currently adopt **Steepst Descent** as our algorithm to compute the gradient. Finally it evaluates the network on the test dataset and carry out the classification accuracy.
     
Intel Xeon (multi-core) & Intel Xeon Phi (many-core) platforms share the same code base. To run on Xeon Phi, compile the program with `'-mmic'` compiler option. More details will be presented in the Compiling section.
     
To get a better performance on Intel Xeon Phi platform, one can change the hard-coded OpenMP parameters in the source code and `consts.h`.
     
### About the algorithm
     
By following the deep learning [UFLDL Tutorial](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial) from Standford University, we implemented a Stacked Autoencoder neural network under the framework provided by the tutorial in Matlab language. Then we ported the Matlab program into pure C/C++ program using MKL and OpenMP technology. If you are interested in our Matlab code, welcome to contact us.
     
For a further detailed description of our work and the performance experiment results, please see our paper titled [Training Large Scale Deep Neural Networks on the Intel Xeon Phi Many-core Coprocessor](http://pasa-bigdata.nju.edu.cn/people/ronggu/pub/DeepLearning_ParLearning.pdf) which is published in Proc. of the 2014 IEEE 28th International Parallel & Distributed Processing Symposium Workshops(ParLearning 2014).
     
     
## Compiling
     
### Requirements
     
In order to compile the program, one needs to install the *Intel C/C++ Compiler* and *Intel MKL Library* first.
     
Intel C/C++ Compiler and MKL should be configured that can be called directly in the command line environment. For example, 'icc/icpc' can be directly called in the command line and *icc* can find the MKL library when the compiler option '-mkl' is presented. To use user specific compiling style, please change the `Makefile`
     
To compile for the Xeon Phi platform, MIC Platform Software Stack (MPSS) should also be installed.
     
### Compile
     
Run `make`. It will try to compile for both multi-core & Xeon Phi(many-core) platform.
     
The executable file for multi-core platform is 'stackedAutoEncoder' and the file for Xeon Phi is 'stackedAutoEncoder.mic'.
     
A subset of MNIST dataset is also provided with the source code. You can try to compile the code and run on that dataset directly without changing anything in the source code. For more details, please see the 'Run' section.
     
     
## Configuration
     
The default configuration will run on an example training set file `train5.txt` & `trainLabels5.txt` and test set file `test.txt` & `testLabels.txt`. The example data set is compressed and placed under the source code directory as `dataset.tar.gz`. Training set file consists of 5000 samples with 784 features. The test set file consists of 10000 test samples. The example data set is a subset from MNIST data set in the formation required by our program. Users can use their own training set and test set files.
     
The following subsections will introduce how to use your own data sets.
     
### File Formation
     
There are four files that should be prepared by the users:
1. Train Data Set File
2. Train Label File
3. Test Data Set File
4. Test Label File
     
**Train Data Set File** Train Data Set File is a text file which contains a matrix that describe the train data set. The matrix stored in the file is a M*N matrix where M is the length of the feature vector of each sample and N is the number of samples. Each column in the matrix is a feature vector of a sample. For example, if the train data set file is `train.txt` which is shown below:
     
    train.txt
    -----------------------
    1 2 3 4 5
    0 1 2 3 4
    7 8 9 0 1
    
Then it represents a train data set that have 5 samples and each sample has a feature vector that contains 3 elements.
Numbers in the matrix are separated by the space ' '.
**Notice** There is *no* train label column in the train data set matrix.
     
**Train Label File** Train Label File consists of labels of training data set. The file consists of N lines where N is the number of samples. Each line is a number which represents the label of the corresponding sample. If there are 10 classes in total, the range of the label is between 1 to 10. For the example before, if there are 10 classes, then the train label file may be :
     
    trainLabel.txt
    ------------------------
    1
    5
    10
    2
    3
    
**Test Data Set File & Label File** Test data set file and label file have the same format as train data set files.
     
     
### How to configure
Currently all the paths and parameters of the network are hard-coded in the source code file `main.cpp`. So users need to change the following sections to customize the data set and network configuration.
     
    //////////About train & test data set
    const int FEATURE_NUMBER = 784;//how many features does a sample have?
    const int CLASS_NUMBER = 10;// how many classes are there? The number of classes in the classification task
    const char* TRAIN_FILE_PATH = "train5.txt";
    const char* TRAIN_LABEL_FILE_PATH = "trainLabels5.txt";
    const int TRAIN_SAMPLE_NUMBER = 5000;//how many samples are there in the train set?
    const char* TEST_FILE_PATH = "test.txt";
    const char* TEST_LABEL_FILE_PATH = "testLabels.txt";
    const int TEST_SAMPLE_NUMBER = 10000;//how many samples are there in the test set?
    //////////About Network
    //Set the hidden layer size info of the Deep Neural Network.
    const int LAYER_DEPTH = 2;//how many hidden layers?
    int layerInfo[LAYER_DEPTH] = {200,200}; //how many nodes are there in each hidden layer?
    
Some extra parameters such as learning rate is hard-coded in the `main.cpp`. If users want to change that, please change it directly in the source code file.

## Run
     
### Run on Multi-core platform
     
The executable file is named `stackedAutoEncoder`. The MKL library is dynamicly linked, so before running the program, please use `ldd` program to check whether the operating system can find all the libraries. Run MKL environment configuration shell scripts if necessary.
     
Place the example data set files(train5.txt, trainLabels5.txt, test.txt, testLabels.txt) with the executable file `stackedAutoEncoder` under the same directory. Then you can test the program with the default configuration by running `./stackedAutoEncoder` or `make run` directly under that directory. On our platform we got the following result:
     
    before fine tune
    the accuracy is: 0.4629
    after fine tune
    the accuracy is: 0.7769
    finish all the procedure
     
If you have changed the configuration,  remember to re-compile the program and place the data set files properly.
     
### Run on Xeon Phi
The executable file is `stackedAutoEncoder.mic` for Xeon Phi. We compile the source code with `-mmic` compiler flag, so the executable file is a **native** executable file for Xeon Phi platform.
     
Please copy all the necessary libraries (for example, *OpenMP* and *MKL* libraries) of MIC architecture to the Xeon Phi card before running.
     
Configure the environment variable **LD_LIBRARY_PATH** if necessary. You can check whether the environment is configured well by running `ldd ./stackedAutoEncoder.mic` in the shell. If everything is OK, then every dynamic link library can be found.
     
Copy the executable file to the Xeon Phi and also copy all the train & test data set and label files to the Xeon Phi card. Put them into proper directory to make sure that your program can find the data set file by the paths you set in the `main.cpp`.
     
Feel free to run the program like the style you run it on multi-core platform :)
     
## Notice ##
All the source code downloaded from here can only be used for study or reasearch. If you want to use it for other purposes, please also contact us(Lei Jin, Zhaokang Wang, [Rong Gu](http://pasa-bigdata.nju.edu.cn/people/ronggu/)) first. 

**Disclaimer**: due to the special situation, we explicitly claims below: we will not be responsible for any issue as a result of mis-use problems here. 

**MNIST Dataset** We provide an formated data set which comes with the source code as an example. The data set is a subset of [MNIST Database](http://yann.lecun.com/exdb/mnist/), please refer to that page before you use the data set.

