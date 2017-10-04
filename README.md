# Semantic Segmentation
### Introduction
The goal of this project is to train a neural network that is able to classify each pixel in an image as either road or non-road. This is known as a Fully Convolutional Network (FCN).

### Development
The first task is to take a regular old VGG neural network, and chop the end off it. We load the VGG network with:

```
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
```

And then the first step of building the decoder is we grab the output of layer 7 of the VGG network, and put a 1x1 convolution on the end of it.

Then to build the rest of the decoder part of the Fully Convolutional Network, we add a conv2d_transpose layer to upsample the output 2x, making sure to also add a regularizer, and a normal random initializer. Without these two things, I was not getting very good cross entropy loss or image output results.

Then we add a skip layer from VGG's layer 3 and add the output from that into the output of the transpose layer. But before we do that we need to do a 1x1 convolution, to get it into the right shape.

Then we do the same again, another conv2d_transpose to upsample 2x, another skip layer, this time from further back, VGG layer 3.

Then we do it one more time, this time upsampling 8x, but no additional skip layer.

Now we have our decoder, and it's time to train it. Our loss is defined as softmax cross entropy, comparing the network's output to the known road-labelled pixel images.

Training it for 10 epochs originally, I got a loss of 1, and inference output labelled images such as:

[](1.png)

I added the normal random initializer to each decoder layer, and got much better results, bringing the loss down to 0.08:

[](2.png)

I increased epochs to 20, batch size to 20 from 10, and retrained, and got even better results, with loss down at 0.04:

[](3.png)

Here are some more sample output images. As you can see, the road is pretty clearly marked.

[](4.png)
[](5.png)
[](6.png)
[](7.png)

A pretty good result for semantic segmentation of the road images!

##### Run
After downloading data.zip and extracting it to the data folder, run the following command to run the project:
```
python3 main.py
```


