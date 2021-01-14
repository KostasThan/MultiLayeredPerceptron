import java.io.*;
import java.util.ArrayList;
import java.util.Random;

import ActivationFunctions.ActivationFunction;
import ActivationFunctions.Logistic;
import ActivationFunctions.TAN;

public class Main {

	// can be used to train another neural network of choice
	private static final String TRAIN_DATA_FILE = ".//resources//train data.txt";
	private static final String TEST_DATA_FILE = ".//resources//test data.txt";
	private static final String OBJECT_FILE = ".//resources//obj";

	// best neural network values
	private static final int INPUT_NODES = 2;
	private static final int FIRST_LAYER_HIDDEN_NEURONS = 11;
	private static final int SECOND_LAYER_HIDDEN_NEURONS = 8;
	private static final int OUTPUT_NODES = 4;
	private static final ActivationFunction[] ACT_FUNCS = new ActivationFunction[] { new TAN(), new Logistic(),
			new Logistic() };
	private static final double learning_rate = 0.001;
	private static final int MIN_EPOCHS = 500;
	private static final int BATCH_SIZE = 3000;
	private static final int MIN_ERROR_DIFFERENCE_BETWEEN_EPOCHS = 5;
	private static final double MEAN_GENERALIZE_ERROR_ACHIEVED = 11.3;

	public static void main(String[] args) throws IOException, InterruptedException {

		// lists to be loaded with the inputs and answers
		ArrayList<double[]> trainInputs = new ArrayList<>();
		ArrayList<double[]> trainAnswers = new ArrayList<>();
		ArrayList<double[]> testInputs = new ArrayList<>();
		ArrayList<double[]> testAnswers = new ArrayList<>();

		// populates the list with the contents of the train data.txt
		populateLists(trainInputs, trainAnswers, TRAIN_DATA_FILE);

		// populate the test data lists
		populateLists(testInputs, testAnswers, TEST_DATA_FILE);

		int[] neuronsPerLayer = new int[] { INPUT_NODES, FIRST_LAYER_HIDDEN_NEURONS, SECOND_LAYER_HIDDEN_NEURONS,
				OUTPUT_NODES };

		// ---Code used to write the best mlp found in disk to be reconstructed for
		// later use
		// if obj file cannot be read in your system
		// create the best found mlp for the current dataset with below code
//        while(true){
//            MLP m = new MLP(neuronsPerLayer,
//                    ACT_FUNCS,learning_rate);
//
//            m.train(trainInputs,trainAnswers,MIN_EPOCHS,BATCH_SIZE,MIN_ERROR_DIFFERENCE_BETWEEN_EPOCHS);
//            if(m.getGeneralizationError(testInputs,testAnswers) < MEAN_GENERALIZE_ERROR_ACHIEVED) {
//                printMLP(m,testInputs,testAnswers);
//                break;
//            }
//
//        }

		// loading best found mlp

		try (ObjectInputStream obj = new ObjectInputStream(new FileInputStream(OBJECT_FILE));) {
			MLP mlp;
			mlp = (MLP) obj.readObject();
			System.out.println("MLP read from the file: ");

			System.out.println(mlp);

			System.out.println("Generalization error: " + mlp.getGeneralizationError(testInputs, testAnswers));

		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}

		// -----------SAMPLE USAGE FOR NEW MLP-----------

		/*
		 * 
		 * 
		 * //-----------------Creating the mlp MLP newMlp = new
		 * MLP(neuronsPerLayer,ACT_FUNCS,learning_rate);
		 * 
		 * //Passing train parameters
		 * newMlp.train(trainInputs,trainAnswers,MIN_EPOCHS,BATCH_SIZE,
		 * MIN_ERROR_DIFFERENCE_BETWEEN_EPOCHS);
		 * 
		 * //Printing generalization error System.out.println("Generalization error: " +
		 * newMlp.getGeneralizationError(testInputs,testAnswers));
		 */

	}

	// -------BELOW FUNCTIONS CONTAINS CODE USED TO PRINT RESULTS, LOAD DATA ETC...

	// print the selected mlp in storage so it loaded later
	// also prints the correct and false anweres in seperate txt folders

	private static void printMLP(MLP mlp, ArrayList<double[]> testInputs, ArrayList<double[]> testAnswers) {

		try (ObjectOutputStream obj = new ObjectOutputStream(new FileOutputStream(OBJECT_FILE));
				BufferedWriter falseWriter = new BufferedWriter(new FileWriter("false.txt"));
				BufferedWriter correctWriter = new BufferedWriter(new FileWriter("correct.txt"));) {

			double[][] guess;
			for (int i = 0; i < testInputs.size(); i++) {
				guess = mlp.forwardPass(testInputs.get(i));

				if (!answeredCorrectly(guess, testAnswers.get(i))) {
					falseWriter.write(testInputs.get(i)[0] + " " + testInputs.get(i)[1] + "\n");
				} else {
					correctWriter.write(testInputs.get(i)[0] + " " + testInputs.get(i)[1] + "\n");
				}

			}
			falseWriter.flush();
			correctWriter.flush();
			obj.writeObject(mlp);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static boolean answeredCorrectly(double[][] guess, double[] answers) {
		double[][] answersVector = MatrixMath.transpose(answers);
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

	private static void populateLists(ArrayList<double[]> inputs, ArrayList<double[]> answers, String filename)
			throws IOException {
		BufferedReader trainDataInput = new BufferedReader(new FileReader(filename));
		String line;
		try (trainDataInput) {
			while ((line = trainDataInput.readLine()) != null && !(line.equals(" "))) {
				String[] contents = line.split(" ");
				double[] inputArray = new double[2];
				inputArray[0] = Double.parseDouble(contents[0]);
				inputArray[1] = Double.parseDouble(contents[1]);

				double[] ansArray;

				if ("c1".equals(contents[2]))
					ansArray = new double[] { 1, 0, 0, 0 };
				else if ("c2".equals(contents[2]))
					ansArray = new double[] { 0, 1, 0, 0 };
				else if ("c3".equals(contents[2]))
					ansArray = new double[] { 0, 0, 1, 0 };
				else
					ansArray = new double[] { 0, 0, 0, 1 };

				inputs.add(inputArray);
				answers.add(ansArray);

			}
		}
	}

	private static double getRandomDouble() {
		Random r = new Random();
		return r.nextDouble() * 2 - 1; // returns a value from -1 to 1;
	}

}
