#include <iostream>
#include <sstream>
#include "util.h"
#include "red.h"

int main(int argc, char *argv[])
{
	unsigned char ** data;
	int rows, columns, size;
	data = readImages("./data/mnist/train-images.idx3-ubyte", size, rows, columns);
	
	unsigned char * labels;
	labels = readLabels("./data/mnist/train-labels.idx1-ubyte");
	
	
	double *** normalizeData = normalize(data, size, rows, columns);
	
	unsigned char ** data1;
	int rows1, columns1, size1;
	data1 = readImages("./data/mnist/t10k-images.idx3-ubyte", size1, rows1, columns1);
	
	unsigned char * labels1;
	labels1 = readLabels("./data/mnist/t10k-labels.idx1-ubyte");
	
	double *** normalizeData1 = normalize(data1, size1, rows1, columns1);
	
	Red red;
	/*
	red.read("6.81");
	red.test(normalizeData1, labels1, size1);
		
	double acierto = ((red.nCorrect)/10000.0) * 100;
	double error = 100-acierto;
	std::cout << "Error: " << error << "%" << std::endl;
	*/
	
	for(int iteracion=0; iteracion<50; iteracion++)
	{
		red.train(normalizeData, labels, size);
		std::cout << "Fin entrenamiento" << std::endl;
		
		red.test(normalizeData1, labels1, size1);
		
		double acierto = ((red.nCorrect)/10000.0) * 100;
		double error = 100-acierto;
		std::cout << "Iteración no " << iteracion << ": " << error << "%" << std::endl;
		std::ostringstream ss;
		ss << error;
		std::string s(ss.str());
		red.save(s);
	}
}