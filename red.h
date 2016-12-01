#ifndef RED
#define RED


#include <iostream>
#include <vector>
#include <cmath>

class Red
{
	
private:
	
	double error;
	// tamaño capa entrada
	static const int inputSize = 28*28;
	// tamaño capa oculta 128
	static const int hiddenSize = 100;
	// tamaño capa salida
	static const int outputSize = 10;
	// tasa de aprendizaje 1e-3
	static const double learning_rate = 0.1;
	static const double momentum = 0.9;
	static const int epochs = 1;
	static const double epsilon = 0.1;
	
	// Image size in MNIST database
	static const int width = 28;
	static const int height = 28;
	
	
	double salidaCapaEntrada[inputSize];
	double salidaCapaOculta[hiddenSize];
	double salidaCapaSalida[outputSize];
	
	double *pesosCapaOculta[inputSize];
	double *pesosCapaSalida[hiddenSize];


	double deltaSalidaCapaSalida[outputSize];
	double deltaCapaOculta[hiddenSize];
	double deltaCapaSalida[outputSize];
	
	double *deltaPesosCapaOculta[inputSize];
	double *deltaPesosCapaSalida[hiddenSize];	

	double expected[outputSize];
	
public:
	
	int nCorrect;
	
	Red()
	{
		/********************************************** 
		 * Reserva de memoria
		 **********************************************/		
		for (int i=0; i<inputSize; i++)
		{
			pesosCapaOculta[i] = new double[hiddenSize];
			deltaPesosCapaOculta[i] = new double[hiddenSize];
		}
		
		for (int i=0; i<hiddenSize; i++) {
			pesosCapaSalida[i] = new double[outputSize];
			deltaPesosCapaSalida[i] = new double[outputSize];
		}
		/**********************************************/
		
		// Inicializar pesos de la capa de entrada a la capa oculta
		for (int i=0; i<inputSize; i++) 
		{
			for (int j=0; j<hiddenSize; j++) 
			{
				int sign = rand() % 2;				
				pesosCapaOculta[i][j] = (double)(rand()/static_cast <float> (RAND_MAX))-0.27;
				if (sign == 1)
					pesosCapaOculta[i][j] = -pesosCapaOculta[i][j];
			}
		}
		
		// Inicializar pesos de la capa oculta a la capa de salida
		for (int j=0; j<hiddenSize; j++) 
		{
			for (int k=0; k<outputSize; k++) 
			{
				int sign = rand() % 2;
				pesosCapaSalida[j][k] = (double)(rand()/static_cast <float> (RAND_MAX))-0.27;
				if (sign == 1)
					pesosCapaSalida[j][k] = -pesosCapaSalida[j][k];
			}
		}
	}
	
	double sigmoid(double x) 
	{
		return 1.0 / (1.0 + exp(-x));
	}
	
	void perceptron() 
	{
		// calculamos los datos de la entrada de la capa oculta con las salidas
		// de la capa de entrada y los pesos de esa capa
		// out1 es la imagen de entrada
		// los pesos se inicializaban aleatoriamente
		// los datos resultantes son las entradas a la capa oculta
		double suma;
		for(int j=0; j<hiddenSize; j++)
		{
			suma = 0.0;
			for (int i=0; i<inputSize; i++)
				suma += pesosCapaOculta[i][j] * salidaCapaEntrada[i];
			suma += 1; // bias
			salidaCapaOculta[j] = sigmoid(suma);
		}

		// los datos de entrada de la capa de salida se calculan con las salidas
		// de la capa oculta y sus pesos
		// los datos de entrada son los de salida por sus pesos
		for (int k=0; k<outputSize; k++)
		{
			suma = 0.0;
			for (int j=0; j<hiddenSize; j++)
				suma += pesosCapaSalida[j][k] * salidaCapaOculta[j];
			suma += 1; // bias
			salidaCapaSalida[k] = sigmoid(suma);
		}
		
	}	
	
	void back_propagation() {
		
		double suma;
		
		for(int k=0; k<outputSize; k++)
			deltaCapaSalida[k] = (expected[k]-salidaCapaSalida[k])*salidaCapaSalida[k]*(1 - salidaCapaSalida[k]);
		
		for(int j=0; j<hiddenSize; j++)
		{
			for(int k=0; k<outputSize; k++)
			{
				deltaPesosCapaSalida[j][k] = learning_rate*deltaCapaSalida[k]*salidaCapaOculta[j] + momentum*deltaPesosCapaSalida[j][k];
				pesosCapaSalida[j][k] += deltaPesosCapaSalida[j][k];
			}
		}

		for(int j=0; j<hiddenSize; j++)
		{
			suma = 0.0;
			for(int k=0; k<outputSize; k++)
				suma += deltaCapaSalida[k]*pesosCapaSalida[j][k];
			deltaCapaOculta[j] = salidaCapaOculta[j]*(1-salidaCapaOculta[j])*suma;
		}
		
		for(int i=0; i<inputSize; i++)
		{
			for(int j=0; j<hiddenSize; j++)
			{
				deltaPesosCapaOculta[i][j] = learning_rate*deltaCapaOculta[j]*salidaCapaEntrada[i] + 
											 momentum*deltaPesosCapaOculta[i][j];
				pesosCapaOculta[i][j] += deltaPesosCapaOculta[i][j];
			}
		}
	}
	
	double square_error()
	{
		double res = 0.0;
		// el error cuadrático es la diferencia entre la salida que tenemos y
		// la salida que se espera al cuadrado
		for (int i=0; i<outputSize; i++)
			res += (salidaCapaSalida[i] - expected[i]) * (salidaCapaSalida[i] - expected[i]);
		
		// lo multiplicamos por 1/2
		res *= 0.5;
		return res;
	}
	
	void learning_process() 
	{
		for (int i=0; i<inputSize; i++) 
		{
			for (int j=0; j<hiddenSize; j++) 
			{
				deltaPesosCapaOculta[i][j] = 0.0;
			}
		}
		
		// Inicializar pesos de la capa oculta a la capa de salida
		for (int j=0; j<hiddenSize; j++) 
		{
			for (int k=0; k<outputSize; k++) 
			{
				deltaPesosCapaSalida[j][k] = 0.0;
			}
		}
		
		for (int i=0; i<epochs; i++) {
			perceptron();
			back_propagation();
			if (square_error() < epsilon) {
				break;
			}
		}
	}
	
	void train(double *** data, unsigned char * labels, int size)
	{
		for(int i=0; i<size; i++) 
		{
			int d[width][height];
			char number;
			for (int j=0; j<width; j++) 
			{
				for (int k=0; k<height; k++) 
				{
					int pos = k + j*width;
					salidaCapaEntrada[pos] = data[i][j][k];
				}
			}
			for (int j=0; j<outputSize; j++) {
				expected[j] = 0.0;
			}
			expected[labels[i]] = 1.0;
			
			// Learning process: Perceptron (Forward procedure) - Back propagation
			learning_process();
		}
	}
	
	void test(double *** data, unsigned char * labels, int size)
	{
		nCorrect=0;
		for(int i=0; i<size; i++)
		{
			int d[width][height];
			//std::cout << "Sample " << i << ": ";
			for (int j=0; j<width; j++) 
			{
				for (int k=0; k<height; k++) 
				{
					int pos = k + j*width;
					salidaCapaEntrada[pos] = data[i][j][k];
				}
			}
			for (int j=0; j<outputSize; j++) {
				expected[j] = 0.0;
			}
			expected[labels[i]] = 1.0;
			
			// Classification - Perceptron procedure
			perceptron();
			
			// Prediction
			int predict = 0;
			for (int j=1; j<outputSize; j++) {
				if (salidaCapaSalida[j] > salidaCapaSalida[predict]) {
					predict = j;
				}
			}
			
			if(predict == labels[i]) nCorrect++;
			//std::cout << predict << " -> " << (int)labels[i] << std::endl;
		}
	}
	
	void save(std::string name)
	{
		std::ofstream out(name.c_str());
		// guardar pesos capa entrada
		for(int i=0; i<inputSize; i++)
		{
			for(int j=0; j<hiddenSize; j++)
			{
				out << pesosCapaOculta[i][j] << " ";
			}
			out << std::endl;
		}
		// guardar pesos capa oculta
		for(int i=0; i<hiddenSize; i++)
		{
			for(int j=0; j<outputSize; j++)
			{
				out << pesosCapaSalida[i][j] << " ";
			}
			out << std::endl;
		}
		
		
		out.close();
		
	}
	
	void read(std::string ruta)
	{
		std::ifstream in(ruta.c_str());
		// leer pesos capa etrada
		for(int i=0; i<inputSize; i++)
		{
			for(int j=0; j<hiddenSize; j++)
			{
				in >> pesosCapaOculta[i][j];
			}
		}
		// leer pesos capa oculta
		for(int i=0; i<hiddenSize; i++)
		{
			for(int j=0; j<outputSize; j++)
			{
				in >> pesosCapaSalida[i][j];
			}
		}
		in.close();
	}
	
	/*
	~Red()
	{
		for(int i=0; i<n; i++)
		{
			for(int k=0; k<rows; k++)
				delete[] plantilla[i][k];
			delete[] plantilla[i];
		}
		delete[] plantilla;
		delete[] results;
	}
	*/
};

#endif