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
	static const int hiddenSize = 128;
	// tamaño capa salida
	static const int outputSize = 10;
	// épocas 512
	static const int epochs = 512;
	// tasa de aprendizaje 1e-3
	static const double learning_rate = 1e-2;
	static const double momentum = 0.9;
	
	static const double epsilon = 1e-2;
	
	// Image size in MNIST database
	static const int width = 28;
	static const int height = 28;
	
	
	// From layer 1 to layer 2. Or: Input layer - Hidden layer
	double *w1[inputSize], *delta1[inputSize], *out1;

	// From layer 2 to layer 3. Or; Hidden layer - Output layer
	double *w2[hiddenSize], *delta2[hiddenSize], *in2, *out2, *theta2;

	// Layer 3 - Output layer
	double *in3, *out3, *theta3;
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
			w1[i] = new double [hiddenSize];
			delta1[i] = new double [hiddenSize];
		}
		
		out1 = new double [inputSize];

		// Layer 2 - Layer 3 = Hidden layer - Output layer
		for (int i=0; i<hiddenSize; i++) {
			w2[i] = new double [outputSize];
			delta2[i] = new double [outputSize];
		}
		
		in2 = new double [hiddenSize];
		out2 = new double [hiddenSize];
		theta2 = new double [hiddenSize];

		// Layer 3 - Output layer
		in3 = new double [outputSize];
		out3 = new double [outputSize];
		theta3 = new double [outputSize];
		/**********************************************/
		
		// Inicializar pesos de la capa de entrada a la capa oculta
		for (int i=0; i<inputSize; i++) 
		{
			for (int j=0; j<hiddenSize; j++) 
			{
				// calculamos el signo del peso aleatoriamente
				int sign = rand() % 2;
				
				// los pesos se inicializan aleatoriamente entre -0.73 y 0.73
				
				w1[i][j] = (double)(rand()/RAND_MAX)-0.27;
				if (sign == 1)
					w1[i][j] = -w1[i][j];
			}
		}
		
		// Inicializar pesos de la capa oculta a la capa de salida
		for (int i=0; i<hiddenSize; i++) 
		{
			for (int j=0; j<outputSize; j++) 
			{
				// calculamos el signo aleatoriamente
				int sign = rand() % 2;
				
				// los pesos se inicializan aleatoriamente entre -1 y 1

				w2[i][j] = (double)(rand()/RAND_MAX);
				if (sign == 1)
					w2[i][j] = -w2[i][j];
			}
		}
	}
	
	double sigmoid(double x) 
	{
		return 1.0 / (1.0 + exp(-x));
	}
	
	void perceptron() 
	{
		// inicializamos la entrada de la capa oculta a 0
		for (int i=0; i<hiddenSize; i++)
			in2[i] = 0.0;

		// inicializamos la entrada de la capa de salida a 0
		for (int i=0; i<outputSize; i++)
			in3[i] = 0.0;

		// calculamos los datos de la entrada de la capa oculta con las salidas
		// de la capa de entrada y los pesos de esa capa
		// out1 es la imagen de entrada
		// los pesos se inicializaban aleatoriamente
		// los datos resultantes son las entradas a la capa oculta
		for (int i=0; i<inputSize; i++)
			for (int j=0; j<hiddenSize; j++)
				in2[j] += out1[i] * w1[i][j];

		// se aplica la función sigmoide
		// out2 son los datos de salida de la capa oculta
		for (int i=0; i<hiddenSize; i++)
			out2[i] = sigmoid(in2[i]);

		// los datos de entrada de la capa de salida se calculan con las salidas
		// de la capa oculta y sus pesos
		// los datos de entrada son los de salida por sus pesos
		for (int i=0; i<hiddenSize; i++)
			for (int j=0; j<outputSize; j++)
				in3[j] += out2[i] * w2[i][j];

		// aplicamos la sigmoide
		// out3 son los datos de salida de la capa de salida (resultados)
		for (int i=0; i<outputSize; i++)
			out3[i] = sigmoid(in3[i]);
	}
	
	// función que calcula el error cuadrático
	double square_error()
	{
		double res = 0.0;
		// el error cuadrático es la diferencia entre la salida que tenemos y
		// la salida que se espera al cuadrado
		for (int i=0; i<outputSize; i++)
			res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
		
		// lo multiplicamos por 1/2
		res *= 0.5;
		return res;
	}
	
	
	void back_propagation() {
		double sum;

		// calculamos el error que comete cada neurona para propagarlo
		
		// calculamos el error
		for (int i=0; i<outputSize; i++) {
			theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
		}

		for (int i=0; i<hiddenSize; i++) {
			sum = 0.0;
			for (int j=0; j<outputSize; j++) {
				sum += w2[i][j] * theta3[j];
			}
			theta2[i] = out2[i] * (1 - out2[i]) * sum;
		}

		for (int i=0; i<hiddenSize; i++) {
			for (int j=0; j<outputSize; j++) {
				delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
				w2[i][j] += delta2[i][j];
			}
		}

		for (int i=0; i<inputSize; i++) {
			for (int j=0; j<hiddenSize; j++) {
				delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
				w1[i][j] += delta1[i][j];
			}
		}
	}
	
	int learning_process() 
	{
		// inicializamos los valores delta
		for (int i=0; i<inputSize; i++) {
			for (int j=0; j<hiddenSize; j++) {
				delta1[i][j] = 0.0;
			}
		}
		// inicializamos los valores delta
		for (int i=0; i<hiddenSize; i++) {
			for (int j=0; j<outputSize; j++) {
				delta2[i][j] = 0.0;
			}
		}

		for (int i=0; i<epochs; i++) {
			perceptron();
			back_propagation();
			if (square_error() < epsilon) {
				return i;
			}
		}
		return epochs;
	}
	
	void train(double *** data, unsigned char * labels, int size)
	{
		for(int i=0; i<size; i++) 
		{
			int d[width][height];
			std::cout << "Sample " << i << std::endl;
			char number;
			for (int j=0; j<width; j++) 
			{
				for (int k=0; k<height; k++) 
				{
					int pos = k + j*width;
					out1[pos] = data[i][j][k];
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
					out1[pos] = data[i][j][k];
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
				if (out3[j] > out3[predict]) {
					predict = j;
				}
			}
			
			if(predict == labels[i]) nCorrect++;
			std::cout << predict << " -> " << (int)labels[i] << std::endl;
		}
	}
	
	void save()
	{
		std::ofstream out("red.txt");
		// guardar pesos capa entrada
		for(int i=0; i<inputSize; i++)
		{
			for(int j=0; j<hiddenSize; j++)
			{
				out << w1[i][j] << " ";
			}
			out << std::endl;
		}
		// guardar pesos capa oculta
		for(int i=0; i<hiddenSize; i++)
		{
			for(int j=0; j<outputSize; j++)
			{
				out << w2[i][j] << " ";
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
				in >> w1[i][j];
			}
		}
		// leer pesos capa oculta
		for(int i=0; i<hiddenSize; i++)
		{
			for(int j=0; j<outputSize; j++)
			{
				in >> w2[i][j];
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