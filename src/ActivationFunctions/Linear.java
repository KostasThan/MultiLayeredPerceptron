package ActivationFunctions;


import java.io.Serializable;

public class Linear implements ActivationFunction, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6156757194317571632L;

	@Override
	public double function(double x) {
		return x;
	}

	@Override
	public double derivativeFromFunctionOutput(double x) {

		return 1;
	}

}
