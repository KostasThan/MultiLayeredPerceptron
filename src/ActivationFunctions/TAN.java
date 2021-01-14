package ActivationFunctions;


import java.io.Serializable;

public class TAN implements ActivationFunction, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4524140353589222823L;

	@Override
	public double function(double x) {
		//cannot implement tanh by hand
		//for large inputs it returns NaN
		//so used the build in function
		return Math.tanh(x);
	}

	@Override
	public double derivativeFromFunctionOutput(double x) {
		return 1 - x * x;
	}

}
