# MultiLayeredPerceptron
This class implemenets a Multi Layered Perceptron.
There is an already predifined classification problem with 4 categories.
Inside the resources file can be found 2 txt files:
 * A file containing the train group with 3000 lines of data and a noise of around 8%.
 * A file containing the test group with 3000 lines of data and no noise.

After some research the best generalization error achieved was around 6% and an obj of mlp
is at the resources folder for you to load and test.

The MLP library is heavily parameterized. You can set parameters for:
 * Batch size
 * Activation function
 * Number of layers
 * Number of neuron per layer

and many more.

**-----USAGE-----**
1. Create any dataset you want. To use the build-in functions:
 * For the train data create a txt file with the data in one row 
  seperated by a space and at the end specify the category "c1","c2","c3","c4" , and optionally add some noise.
 * Create another txt for the answers with or without noise.
2. Create the architecture for the MLP via the constructor
3. Train the model as much as you want
4. Check the generalization error 

_NOTE that the MLP uses arrays to train, so you have to convert each line of the txt into the appropriate format_

_You can use the built in functions or
you can create txt file with different format if you write your own function to load each line as an array_

**Finally** there are 2 images in the resources file picturing the current dataset with the noise
and the best trained MLP guessses.
