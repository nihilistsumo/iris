package com;

import java.io.IOException;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.core.SparseInstance;

public class JavaMLMain {
	public static void main(String[] args){
		IrisClassificationModel ic = new IrisClassificationModel();
		String dataFilePath = "/home/sumanta/Documents/iris_data/iris.data";
		try {
			Classifier c = ic.getClassifier(dataFilePath);
			ic.classify(c, dataFilePath);
			ic.evaluateClassifier(c, dataFilePath);
			ic.crossValidate(c, dataFilePath);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
