#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


unsigned char ** readImages(std::string filename, int &size, int &rows, int &columns)
{

    std::ifstream file(filename.c_str(), std::ios::binary);

    if(file.is_open()) {
        int magicNumber = 0;

        file.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        if(magicNumber != 2051)
		{
			std::cerr << "Error while reading MNIST data from " << filename << std::endl;
			exit(-1);
		}

        file.read((char *)&size, sizeof(size)), size = reverseInt(size);
        file.read((char *)&rows, sizeof(rows)), rows = reverseInt(rows);
        file.read((char *)&columns, sizeof(columns)), columns = reverseInt(columns);
		std::cout << "Reading " << size << " " << rows << "x" << columns << " images..." << std::endl;
		
		
		int image_size = rows * columns;
		unsigned char** _dataset = new unsigned char*[size];
		
		for(int i = 0; i < size; i++) {
            _dataset[i] = new unsigned char[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;

    } else {
        std::cerr << "Cannot open file " << filename << std::endl;
		exit(-1);
    }
}

unsigned char * readLabels(std::string filename)
{

    std::ifstream file(filename.c_str(), std::ios::binary);

    if(file.is_open()) {
        int magicNumber = 0, size = 0;

        file.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        if(magicNumber != 2049) // 0x00000801 == 08 (unsigned byte) + 01 (vector)
		{
			std::cerr << "Error while reading MNIST labels from  " << filename << std::endl;
			exit(-1);
		}

        file.read((char *)&size, sizeof(size)), size = reverseInt(size);	
		
		unsigned char* labels = new unsigned char[size];
		
		for(int i = 0; i < size; i++) {
            file.read((char *)&labels[i], sizeof(labels[i]));
        }
        return labels;

    } else {
        std::cerr << "Cannot open file " << filename << std::endl;
		exit(-1);
    }
}


double ** normalize (unsigned char* image, int rows, int columns)
{
	double **data;
	
	data = new double*[rows];
	for(int k=0; k<rows; k++)
		data[k] = new double[columns];
	
	for (int i=0; i<rows; i++)
		for (int j=0; j<rows; j++)
			data[i][j] = (double) (image[i*columns + j] / 255.0);
	
	return data;
}

double *** normalize (unsigned char** image, int size, int rows, int columns)
{
	double ***data;
	
	data = new double **[size];
	for(int i=0; i<size; i++)
	{
		data[i] = new double*[rows];
		for(int k=0; k<rows; k++)
			data[i][k] = new double[columns];
	}
	
	for(int i=0; i<size; i++)
	{
		for (int j=0; j<rows; j++)
			for (int k=0; k<rows; k++)
				data[i][j][k] = (double) (image[i][j*columns + k] / 255.0);
	}
	
	return data;
}