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
	static const int hiddenSize = 256;
	// tamaño capa salida
	static const int outputSize = 10;
	// tasa de aprendizaje 1e-3
	static const double learning_rate = 0.1;
	static const double momentum = 0.9;
	static const int epochs = 1;
	static const double epsilon = 1;
	
	// Image size in MNIST database
	static const int width = 28;
	static const int height = 28;
	
	
	double *salidaCapaEntrada;
	double *salidaCapaOculta;
	double *salidaCapaSalida;
	
	double **pesosCapaOculta;
	double **pesosCapaSalida;


	double *deltaSalidaCapaSalida;
	double *deltaCapaOculta;
	double *deltaCapaSalida;
	
	double **deltaPesosCapaOculta;
	double **deltaPesosCapaSalida;	

	double expected[outputSize];
	
public:
	
	int nCorrect;
	
	Red()
	{
		/********************************************** 
		 * Reserva de memoria
		 **********************************************/
		pesosCapaOculta = new double*[hiddenSize];
		deltaPesosCapaOculta = new double*[hiddenSize];
		for (int i=0; i<hiddenSize; i++)
		{
			pesosCapaOculta[i] = new double[inputSize];
			deltaPesosCapaOculta[i] = new double[inputSize];
		}
		
		pesosCapaSalida = new double*[outputSize];
		deltaPesosCapaSalida = new double*[outputSize];
		for (int i=0; i<outputSize; i++) {
			pesosCapaSalida[i] = new double[hiddenSize];
			deltaPesosCapaSalida[i] = new double[hiddenSize];
		}
		
		salidaCapaEntrada = new double[inputSize];
		salidaCapaOculta = new double[hiddenSize];
		salidaCapaSalida = new double[outputSize];


		deltaSalidaCapaSalida = new double[outputSize];
		deltaCapaOculta = new double[hiddenSize];
		deltaCapaSalida = new double[outputSize];
		/**********************************************/
		
		// Inicializar pesos de la capa de entrada a la capa oculta
		for (int j=0; j<hiddenSize; j++) 
		{
			for (int i=0; i<inputSize; i++) 
			{
				int sign = rand() % 2;				
				pesosCapaOculta[j][i] = (double)(rand()/static_cast <float> (RAND_MAX))*0.1;
				if (sign == 1)
					pesosCapaOculta[j][i] = -pesosCapaOculta[j][i];
			}
		}
		
		// Inicializar pesos de la capa oculta a la capa de salida
		for (int k=0; k<outputSize; k++) 
		{
			for (int j=0; j<hiddenSize; j++) 
			{
				int sign = rand() % 2;
				pesosCapaSalida[k][j] = (double)(rand()/static_cast <float> (RAND_MAX))*0.1;
				if (sign == 1)
					pesosCapaSalida[k][j] = -pesosCapaSalida[k][j];
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
				suma += pesosCapaOculta[j][i] * salidaCapaEntrada[i];
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
				suma += pesosCapaSalida[k][j] * salidaCapaOculta[j];
			suma += 1; // bias
			salidaCapaSalida[k] = sigmoid(suma);
		}
		
	}	
	
	void back_propagation() {
		
		for(int k=0; k<outputSize; k++)
		{
			deltaCapaSalida[k] = (expected[k]-salidaCapaSalida[k])*salidaCapaSalida[k]*(1 - salidaCapaSalida[k]);
			for(int j=0; j<hiddenSize; j++)
			{
				deltaPesosCapaSalida[k][j] = learning_rate*deltaCapaSalida[k]*salidaCapaOculta[j] + momentum*deltaPesosCapaSalida[k][j];
				pesosCapaSalida[k][j] += deltaPesosCapaSalida[k][j];
			}
		}
		
		double suma;
		for(int j=0; j<hiddenSize; j++)
		{
			suma = 0.0;
			for(int k=0; k<outputSize; k++)
				suma += deltaCapaSalida[k]*pesosCapaSalida[k][j];
			deltaCapaOculta[j] = salidaCapaOculta[j]*(1-salidaCapaOculta[j])*suma;
			for(int i=0; i<inputSize; i++)
			{
				deltaPesosCapaOculta[j][i] = learning_rate*deltaCapaOculta[j]*salidaCapaEntrada[i] + 
											 momentum*deltaPesosCapaOculta[j][i];
				pesosCapaOculta[j][i] += deltaPesosCapaOculta[j][i];
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
		for (int j=0; j<hiddenSize; j++) 
		{
			for (int i=0; i<inputSize; i++) 
			{
				deltaPesosCapaOculta[j][i] = 0.0;
			}
		}
		
		// Inicializar pesos de la capa oculta a la capa de salida
		for (int k=0; k<outputSize; k++) 
		{
			for (int j=0; j<hiddenSize; j++) 
			{
				deltaPesosCapaSalida[k][j] = 0.0;
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
			std::cout << "Sample " << i << ": " << std::endl;
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
			std::cout << "Sample " << i << ": ";
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
			std::cout << "Sample: " << i << std::endl;
			
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
		for(int j=0; j<hiddenSize; j++)
		{
			for(int i=0; i<inputSize; i++)
			{
				out << pesosCapaOculta[j][i] << " ";
			}
			out << std::endl;
		}
		// guardar pesos capa oculta
		for(int k=0; k<outputSize; k++)
		{
			for(int j=0; j<hiddenSize; j++)
			{
				out << pesosCapaSalida[k][j] << " ";
			}
			out << std::endl;
		}
		
		
		out.close();
		
	}
	
	void read(std::string ruta)
	{
		std::ifstream in(ruta.c_str());
		// leer pesos capa etrada
		for(int j=0; j<hiddenSize; j++)
		{
			for(int i=0; i<inputSize; i++)
			{
				in >> pesosCapaOculta[j][i];
			}
		}
		// leer pesos capa oculta
		for(int k=0; k<outputSize; k++)
		{
			for(int j=0; j<hiddenSize; j++)
			{
				in >> pesosCapaSalida[k][j];
			}
		}
		in.close();
	}
	
	
	~Red()
	{
		for (int j=0; j<hiddenSize; j++)
		{
			delete[] pesosCapaOculta[j];
			delete[] deltaPesosCapaOculta[j];
		}
		delete[] pesosCapaOculta;
		delete[] deltaPesosCapaOculta;
		
		
		for (int k=0; k<outputSize; k++) {
			delete[] pesosCapaSalida[k];
			delete[] deltaPesosCapaSalida[k];
		}
		delete[] pesosCapaSalida;
		delete[] deltaPesosCapaSalida;
		
		delete[] salidaCapaEntrada;
		delete[] salidaCapaOculta;
		delete[] salidaCapaSalida;


		delete[] deltaSalidaCapaSalida;
		delete[] deltaCapaOculta;
		delete[] deltaCapaSalida;
	}
	
};

#endif