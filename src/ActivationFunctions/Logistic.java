package ActivationFunctions;

import java.io.Serializable;

public class Logistic implements ActivationFunction, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4385603068956077309L;

	@Override
	public double function(double x) {
		return  1 / ((1 + Math.exp(-x)));
	}

	@Override
	public double derivativeFromFunctionOutput(double x) {
		return x * (1 - x);
	}

}
