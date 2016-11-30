

all: main


main.o: main.cpp util.h red.h

	g++ -c -o main.o main.cpp -O3

main: main.o
	
	g++ -o main main.o -O3
	
clean:

	rm *.o
	
