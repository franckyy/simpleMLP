package com.franckyy.simpleMLP;

public class SimpleMLP {
	private Double[][] weightsInputHidden;	//poids entrée -> caché
	private Double[][] weightsHiddenOutput;	//poids caché -> sortie
	
	//constructeur : initialise les poids avec des valeurs fixes pour tester
	public SimpleMLP() {
		//2 entrées -> 2 neurones cachés
		weightsInputHidden = new Double[][] {{0.15, 0.25}, {0.20, 0.30}};
		//2 neurones cachés -> 1 sortie
		weightsHiddenOutput = new Double[][] {{0.40}, {0.45}};
	}
	
	//Propagation avant (forward pass)
	public double predict(double[] inputs) {
		//Vérifie que le nombre d'entrées correspond
		if(inputs.length != weightsInputHidden.length) {
			throw new IllegalArgumentException("Nombre d'arguments incorrect");
		}
		
		//Etape 1 : Calcul de la couche cachée
		double[] hidden = new double[weightsInputHidden[0].length];
		for(int i = 0; i < hidden.length; i++) {
			double sum = 0;
			for(int j = 0; j < inputs.length; j++) {
				sum += inputs[j] * weightsInputHidden[j][i];
			}
			hidden[i] = sigmoid(sum);	//activation sigmoide
		}
		
		//Etape 2 : Calcul de la sortie
		double output = 0;
		for(int i = 0; i < hidden.length; i++) {
			output += hidden[i] * weightsHiddenOutput[i][0];
		}
		return sigmoid(output);
	}
	
	private double sigmoid(double x) {
		return 1.0/(1.0 + Math.exp(-x));
	}
}
