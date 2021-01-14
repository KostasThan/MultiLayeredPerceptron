
import java.io.*;
import java.util.*;

import ActivationFunctions.ActivationFunction;

public class MLP implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1496884530626739995L;
	private List<double[][]> weightsList;

	private ActivationFunction[] activationFuncs;
	private List<double[][]> biasesList;
	private double learningRate;
	private ArrayList<double[][]> layerOX;
	private ArrayList<double[][]> weightsDeltaList;
	private ArrayList<double[][]> biasDeltaList;

	/**
	 * @param neuronsPerLayer </br>
	 * The first entry of the array is the number of input nodes thus
	 * when training the model the input size must that of the input nodes. 
	 * </br>
	 * The last element of the array is the output nodes. The result of any
	 * guess of the MLP is going to be a "vertical 2D" array that has as many rows as
	 * the the last element has columns, and 1 column. </br>
	 * </br>
	 * @param activationFunctions passed to the constructor are mapped to the correspondent
	 * layer. E.G First activation function to the first layer 
	 * 
	 * @param learningRate </br>
	 * usually is set to a small value like 0.1 depending on the problem.</br></br>
	 * 
	 * 
	 * <b>Exception Cases:</b></br>
	 * <li>Activation functions array should have a length as neuronsPerLayer array -
	 * 1.
	 * <li>The neuronsPerLayer should at least have a length of 3.Number of neurons for
	 * input layer,for the hidden layer and for the output layer.
	 * 
	 *
	 */
	public MLP(int[] neuronsPerLayer, ActivationFunction[] activationFunctions, double learningRate) {
		if(learningRate<0)
			throw new IllegalArgumentException();
		
		this.learningRate = learningRate;
		weightsList = new ArrayList<>();
		biasesList = new ArrayList<>();
		layerOX = new ArrayList<>();
		weightsDeltaList = new ArrayList<>();
		biasDeltaList = new ArrayList<>();
		initializeWeightsAndBiases(neuronsPerLayer);
		this.activationFuncs = Arrays.copyOf(activationFunctions, activationFunctions.length);
		initializeDeltaLists();
	}

	private void randomizeWeightsAndBiasesAtLayer(int[] neuronsPerLayer, int layer) {
		double[][] weightsArray;
		double[][] biasesArray;

		int neuronsAtCurrentLayer = neuronsPerLayer[layer];
		int neuronsAtPreviousLayer = neuronsPerLayer[layer - 1];

		weightsArray = new double[neuronsAtCurrentLayer][neuronsAtPreviousLayer];
		MatrixMath.randomize(weightsArray);
		weightsList.add(weightsArray);

		// we make it 2D so we can add it later on
		biasesArray = new double[neuronsAtCurrentLayer][1];
		MatrixMath.randomize(biasesArray);
		biasesList.add(biasesArray);
	}

	public double[][] forwardPass(double[] inputs) {

		// inputs check
		if (inputs.length != getNumOfInputNodes()) {
			System.err.println("Wrong numbers of inputs");
			throw new RuntimeException();
		}

		layerOX.clear();

		// converting the inputs to the correct format
		// so they can be multiplied
		double[][] ox = MatrixMath.transpose(inputs);

		// the inputs transposed is the ox of the input layer
		layerOX.add(ox);

		// for every layer in our list
		for (int i = 0; i < weightsList.size(); i++) {

			// get the ux of this layer(first multiply they weights and the add the biases)
			double[][] ux = MatrixMath.add(MatrixMath.multiply(weightsList.get(i), ox), biasesList.get(i));

			// get the activation func from the array
			ActivationFunction currActivationFunc = activationFuncs[i];

			// pass through activation function
			ox = MatrixMath.mapMatrix(ux, (x) -> currActivationFunc.function(x));

			layerOX.add(ox);
		}
		return ox;

	}

	/**
	 *
	 * @param inputs             the train data as 1d array that matches the size of input nodes
	 * @param answers            the answers for the train data as 1d array that matches the size of output nodes
	 * @param minEpochs          the minimum epochs to train the MLP for. 
	 * @param batches            the sizes of the train data before changing the
	 *                           weights
	 * @param minErrorDifference the minimum error difference between two epochs</br>
	 *                           e.g. i want each epoch to learn at least
	 *                           20 extra train data
	 */
	public void train(ArrayList<double[]> inputs, ArrayList<double[]> answers, int minEpochs, int batches,
			double minErrorDifference) {
		// set delta lists to all 0 for the new training session
		initializeDeltaLists();

		// we are at the first epoch. we assume it's all wrong
		int previousEpochError = inputs.size();
		int currentEpoch = 0;

		while (true) {

			int currentEpochErrors = 0;
			for (int i = 0; i < inputs.size(); i++) {

				if (((i + 1) % batches == 0 && batches <= i + 1) || batches == 1) {
					addDeltasToWeightsAndBiases();
					initializeDeltaLists();

				}

				double[][] answersAsVector = MatrixMath.transpose(answers.get(i));
				double[][] guess = forwardPass(inputs.get(i));

				if (!guessedCorrectly(guess, answersAsVector)) {
					currentEpochErrors++;
					ArrayList<double[][]> totalLayerErrors = backPropagate(guess, answersAsVector);
					calculateDeltaWeightsAndBiases(totalLayerErrors);
				}
			}
			System.out.println("Current epoch:" + currentEpoch + ", Errors: " + currentEpochErrors);
			currentEpoch++;
			if (currentEpoch >= 300 && minErrorDifference > (previousEpochError - currentEpochErrors)) {
				previousEpochError = currentEpochErrors;
				break;
			}
			previousEpochError = currentEpochErrors;
		}
	}

	private void calculateDeltaWeightsAndBiases(ArrayList<double[][]> totalLayerErrors) {

		// for every error calculated(which means for every layer)
		for (int i = totalLayerErrors.size() - 1; i >= 0; i--) {

			// calculate gradient
			double[][] gradient = MatrixMath.mapMatrix(layerOX.get(i + 1),
					activationFuncs[i]::derivativeFromFunctionOutput);

			gradient = MatrixMath.hadamardProduct(gradient, totalLayerErrors.get(i));
			gradient = MatrixMath.multiply(gradient, learningRate);

			// calculate deltas
			double[][] currentWeightDelta = MatrixMath.transpose(layerOX.get(i));
			currentWeightDelta = MatrixMath.multiply(gradient, currentWeightDelta);

			// add the to the specific list/index
			MatrixMath.addToArray(weightsDeltaList.get(i), currentWeightDelta);
			MatrixMath.addToArray(biasDeltaList.get(i), gradient);

		}
	}

	private ArrayList<double[][]> backPropagate(double[][] guess, double[][] answersVector) {
		// initializing a list
		// to hold the errors for each layer as a vector(1 column array)
		ArrayList<double[][]> totalLayerErrors = new ArrayList<>();

		// the last layer's error is the answer - output

		double[][] lastLayerError = MatrixMath.subtract(answersVector, guess);

		totalLayerErrors.add(lastLayerError);

		// for every layer's weights (except the first)
		for (int i = weightsList.size() - 1; i >= 1; i--) {

			// we transposed the weights array so we can multiply it
			double[][] weightsTransposed = MatrixMath.transpose(weightsList.get(i));

			// compute the current layer's error by multiplying with the previous layer
			// error
			double[][] currentLayerError = MatrixMath.multiply(weightsTransposed, totalLayerErrors.get(0));

			// since we go from end to start, we add every array to the head of the list
			totalLayerErrors.add(0, currentLayerError);
		}
		return totalLayerErrors;
	}

	private boolean guessedCorrectly(double[][] guess, double[][] answersVector) {

		// getting the pos of the maximum value
		double max = guess[0][0];
		int maxPos = 0;

		// it will always be 1 column 2d array
		for (int i = 1; i < guess.length; i++) {
			if (guess[i][0] > max) {
				max = guess[i][0];
				maxPos = i;
			}
		}
		// the max pos of the guess must be the pos where the answer is 1
		return answersVector[maxPos][0] == 1;
	}

	private void addDeltasToWeightsAndBiases() {
		for (int i = 0; i < weightsList.size(); i++) {
			MatrixMath.addToArray(weightsList.get(i), weightsDeltaList.get(i));
			MatrixMath.addToArray(biasesList.get(i), biasDeltaList.get(i));
		}
	}

	private int getNumOfInputNodes() {
		return weightsList.get(0)[0].length;
	}

	private void initializeDeltaLists() {
		// clearing any previous stuff
		weightsDeltaList.clear();
		biasDeltaList.clear();

		// adding the correct double[][] arrays to each index
		for (int i = 0; i < weightsList.size(); i++) {

			// calculate this layer's weights array size
			int weightRows = weightsList.get(i).length;
			int weightCols = weightsList.get(i)[0].length;
			double[][] weightDelta = new double[weightRows][weightCols];
			weightsDeltaList.add(weightDelta);

			// calculate this layer's bias array size
			int biasRows = biasesList.get(i).length;
			int biasCols = biasesList.get(i)[0].length;
			double[][] biasDelta = new double[biasRows][biasCols];
			biasDeltaList.add(biasDelta);
		}
	}

	
	/**
	 * Return the generalization error based on the inputs-answers provided.
	 * This function <u>does not train the MLP</u> and is not stochastic.</br> For the same
	 * inputs-answers and without training even further the results will always be the same.
	 * @param inputs the inputs to get the generalization error from
	 * @param answers the answers to check if the MLP guessed correctly
	 * @return a double which represents the errors made in the form of a percentage
	 */
	public double getGeneralizationError(ArrayList<double[]> inputs, ArrayList<double[]> answers) {
		double[][] guess;
		double[][] currentAnswer;
		double generalizationError = 0;
		for (int i = 0; i < inputs.size(); i++) {
			guess = forwardPass(inputs.get(i));
			currentAnswer = MatrixMath.transpose(answers.get(i));
			if (!guessedCorrectly(guess, currentAnswer)) {
				generalizationError++;
			}
		}
		return generalizationError * 100 / (inputs.size());

	}

	private void initializeWeightsAndBiases(int[] neuronsPerLayer) {
		if (neuronsPerLayer.length < 3) {
			throw new RuntimeException("You must enter the numbers of neurons for at least three layers\n"
					+ "Layer 1: Input layer " + "Layer 2: First hidden layer" + "Layer 3: Output layer");
		}

		for (int layer = 1; layer < neuronsPerLayer.length; layer++) {
			randomizeWeightsAndBiasesAtLayer(neuronsPerLayer, layer);
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("NeuralNetwork numOfInputNodes=" + getNumOfInputNodes() + "\n");
		for (int i = 0; i < weightsList.size(); i++) {
			sb.append("\t\tRow: " + i + "\nWeigths: \n");
			sb.append(MatrixMath.arrayToString(weightsList.get(i)));
			sb.append("Biases: \n" + MatrixMath.arrayToString(biasesList.get(i)) + "\n");
		}

		return sb.toString();

	}
}
