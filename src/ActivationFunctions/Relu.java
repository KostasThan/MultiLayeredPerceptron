package ActivationFunctions;


import java.io.Serializable;

public class Relu implements ActivationFunction, Serializable {
    /**
	 * 
	 */
	private static final long serialVersionUID = -39257308607609069L;

	@Override
    public double function(double x) {
        return Math.max(0,x);
    }

    @Override
    public double derivativeFromFunctionOutput(double x) {
        return x>0? 1: 0;
    }
}
