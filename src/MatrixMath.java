

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public class MatrixMath {

    public static double[][] makeArray2D(double[] array) {
        double[][] newArray = new double[1][array.length];
        for (int i = 0; i < array.length; i++) {
            newArray[0][i] = array[i];
        }

        return newArray;
    }

    public static double[][] multiply(double[][] a, double[][] b) {

        if (MatrixMath.areMultiplicationEligible(a, b)) {
            int newArrayRows = a.length;
            int newArrayColums = b[0].length;
            double firstArrayColumns = a[0].length;
            double[][] newArray = new double[newArrayRows][newArrayColums];

            for (int i = 0; i < newArrayRows; i++) {

                for (int j = 0; j < newArrayColums; j++) {
                    double sum = 0;
                    for (int k = 0; k < firstArrayColumns; k++) {
                        sum += a[i][k] * b[k][j];
                    }

                    newArray[i][j] = sum;
                }

            }
            return newArray;
        }
        System.err.println("CANNOT MULTIPLY THOSE MATRICES");
        throw new RuntimeException();

    }

    public static double[][] hadamardProduct(double[][] a,double[][] b){
        for(int i = 0; i < a.length; i++){
            for (int j = 0; j < a[i].length; j++){
                a[i][j] = a[i][j] * b[i][j];
            }
        }
        return a;
    }

    //scalar multiplication
    public static double[][] multiply(double[][] a, double scalar){
        return mapMatrix(a,(e) -> e * scalar);

    }

    public static double[][] transpose(double[][] array) {
        int newArrayRows = array[0].length;
        int newArrayColums = array.length;
        double[][] newArray = new double[newArrayRows][newArrayColums];
        for (int i = 0; i < newArrayColums; i++) {
            for (int j = 0; j < newArrayRows; j++) {
                newArray[j][i] = array[i][j];
            }
        }

        return newArray;
    }

    public static double[][] transpose(double[] array) {
        int newArrayRows = array.length;
        int newArrayColums = 1;
        double[][] newArray = new double[newArrayRows][newArrayColums];
        for (int i = 0; i < newArrayRows; i++) {
            newArray[i][0] = array[i];
        }

        return newArray;
    }


    //!!Modifies given array
    public static double[][] addToArray(double[][] a, double[][] b) {

        if (MatrixMath.haveSameDimensions(a, b)) {
            int rows = a.length;
            int columns = a[0].length;

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    a[i][j] += b[i][j];
                }
            }
            return a;
        } else {
            throw new NullPointerException("CANNOT ADD THOSE MATRICES");
        }
    }

    public static double[][] subtractFromArray(double[][] a, double[][] b) {
        if (MatrixMath.haveSameDimensions(a, b)) {
            int rows = a.length;
            int columns = a[0].length;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    a[i][j] -= b[i][j];
                }
            }
            return a;
        } else {
            throw new NullPointerException("CANNOT SUBTRACT THOSE MATRICES");
        }

    }

    public static double[][] subtract(double[][] a, double[][] b) {

        if (MatrixMath.haveSameDimensions(a, b)) {
            int rows = a.length;
            int columns = a[0].length;
            double[][] newArray = new double[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    newArray[i][j] = a[i][j] - b[i][j];
                }
            }
            return newArray;
        } else {
            throw new NullPointerException("CANNOT SUBTRACT THOSE MATRICES");
        }
    }

    public static double[][] add(double[][] a, double[][] b) {

        if (MatrixMath.haveSameDimensions(a, b)) {
            int rows = a.length;
            int columns = a[0].length;
            double[][] newArray = new double[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    newArray[i][j] = a[i][j] + b[i][j];
                }
            }
            return newArray;
        } else {
            throw new NullPointerException("CANNOT ADD THOSE MATRICES");
        }
    }

    public static double[][] mapMatrix(double[][] a, Function<Double, Double> func) {
        int rows = a.length;
        int columns = a[0].length;
        double[][] output = new double [rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                output[i][j] = func.apply(a[i][j]);
            }
        }

        return output;
    }

    public static double[][] add(double[][] a, double num) {

        return mapMatrix(a, (d) -> d + num);
    }

    public static double[][] subtract(double[][] a, double num) {

        return mapMatrix(a, (d) -> d - num);
    }

    public static void randomize(double[][] a) {
        Random rand = new Random();
        int rows = a.length;
        int columns = a[0].length;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {

                a[i][j] = rand.nextDouble() * 2 - 1;
            }
        }
    }

    public static Double getSum(double[][] a) {
        return Arrays.stream(a) // stream of arrays
                .flatMapToDouble(Arrays::stream) // stream of doubles
                .reduce(0, Double::sum); // add them

    }

    public static Double getRowSum(double[][] a, int row) {
        return Arrays.stream(a[row]).reduce(0, Double::sum);

    }

    public static String arrayToString(double[][] a) {
        StringBuilder sb = new StringBuilder();
        Arrays.stream(a).forEach((s) -> sb.append(Arrays.toString(s) + "\n"));
        return (sb.toString());

    }

    private static boolean haveSameDimensions(double[][] a, double[][] b) {
        return a.length == b.length && a[0].length == b[0].length;
    }

    private static boolean areMultiplicationEligible(double[][] a, double[][] b) {

        return a[0].length == b.length;
    }


}
