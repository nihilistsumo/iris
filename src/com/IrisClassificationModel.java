package com;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.evaluation.CrossValidation;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.clustering.Clusterer;
import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

public class IrisClassificationModel {
	public Classifier getClassifier(String datafile) throws IOException{
		Dataset data = FileHandler.loadDataset(new File(datafile), 4, ",");
		Classifier knn = new KNearestNeighbors(5);
		knn.buildClassifier(data);
		return knn;
	}
	public void classify(Classifier c, String datafile) throws IOException{
		Dataset data = FileHandler.loadDataset(new File(datafile), 4, ",");
		for(Instance ins:data){
			System.out.println(ins+" of ID "+ins.getID()+" belongs to class "+c.classify(ins));
		}
	}
	public void evaluateClassifier(Classifier c, String datafile) throws IOException{
		Dataset data = FileHandler.loadDataset(new File(datafile), 4, ",");
		Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(c, data);
		System.out.println("\nAccuracy evaluation of the classifier");
		for(Object o:pm.keySet())
		    System.out.println("Class name: "+o+", Accuracy: "+pm.get(o).getAccuracy());
	}
	public void crossValidate(Classifier c, String datafile) throws IOException{
		CrossValidation cv = new CrossValidation(c);
		Dataset data = FileHandler.loadDataset(new File(datafile), 4, ",");
		Map<Object, PerformanceMeasure> p = cv.crossValidation(data);
		System.out.println("\nCross validation of the classifier");
		for(Object o:p.keySet())
		    System.out.println("Class name: "+o+", Accuracy: "+p.get(o).getAccuracy());
	}
}
