#include <iostream>
#include "util.h"
#include "red.h"

int main(int argc, char *argv[])
{
	unsigned char ** data;
	int rows, columns, size;
	data = readImages("/home/juanpi/Dropbox/MII/IC/P1/data/mnist/train-images.idx3-ubyte", size, rows, columns);
	
	unsigned char * labels;
	labels = readLabels("/home/juanpi/Dropbox/MII/IC/P1/data/mnist/train-labels.idx1-ubyte");
	
	
	double *** normalizeData = normalize(data, size, rows, columns);
	
	Red red;
	
	red.train(normalizeData, labels, size);
	//red.read("red.txt (95.53)");
	
	unsigned char ** data1;
	int rows1, columns1, size1;
	data1 = readImages("/home/juanpi/Dropbox/MII/IC/P1/data/mnist/t10k-images.idx3-ubyte", size1, rows1, columns1);
	
	unsigned char * labels1;
	labels1 = readLabels("/home/juanpi/Dropbox/MII/IC/P1/data/mnist/t10k-labels.idx1-ubyte");
	
	double *** normalizeData1 = normalize(data1, size1, rows1, columns1);
	
	red.test(normalizeData1, labels1, size1);
	
	double acierto = ((red.nCorrect)/10000.0) * 100;
	double error = 100-acierto;
	std::cout << "Acierto: " << acierto << "%" << std::endl;
	std::cout << "Error: " << error << "%" << std::endl;
	
	red.save();
}