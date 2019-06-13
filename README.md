# Witcher character classifier
Convolutional Neural Network (CNN) analyzing the features of characters from the witcher 3.

## Preprocessing
  * Creating a dataset using [Microsoft’s Bing Image Search API](https://azure.microsoft.com/en-us/services/cognitive-services/bing-image-search-api/)
  * Resizing dataset to a size of 224x224
  * Data augmentation techniques:
    * Random rotation
    * Random noise
    * Horizontal flip
    * Random erasing
    
## Transfer learning

I used transfer learning with pre-trained weights from the VGG-16 model. This helped the model to learn better.

### Training process

The model was trained on 100 epochs. It uses the Adam optimizer for gradient descent.

Usage: python src/train_nn.py

  -m, --model              path to trained model model (default: output/witcher-classifier.h5  
  -d, --dataset            path to input dataset (default: dataset)  
  -l, --labelbin           path to output label binarizer (default: lb.pickle)  
  -p, --plot               path to output accuracy/loss plot (default: output/plot.png)  
  --show_examples          show some training examples

## Model loss/accuracy

* Accuracy and Loss:

   ![Image](./output/plot.png) 
   
## User interface

A user interface made with pyqt to make the classification easier:

Usage: python src/run_nn.py

  -m, --model              path to trained model model (default: output/witcher-classifier.h5  
  -l, --labelbin           path to output label binarizer (default: lb.pickle)  


## Possible improvements
  * Using classification with object recognition to detect every characters in the image:
  
    ![Image](./doc/images/object-recognition.jpg)
    
## Authors and acknowledgment
Many thanks to Adrian Rosebrock from pysearchimage for his series of articles:

* [How to (quickly) build a deep learning image dataset](https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/)
* [Keras and Convolutional Neural Networks (CNNs)](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/)

## License
[MIT](https://choosealicense.com/licenses/mit/)
