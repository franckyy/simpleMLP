package com.franckyy.simpleMLP;

public class Main {
	public static void main(String[] args) {
		//Crée une instance du MLP
		SimpleMLP mlp = new SimpleMLP();
		
		//Test avec des entrées simples
		double[] inputs = {1.0, 0.0};	// exemple : [1, 0]
		double output = mlp.predict(inputs);
		
		System.out.println("Entrées : [" + inputs[0] + ", " + inputs[1] + "]");
		System.out.println("Sortie prédite : " + output);
	}
}
