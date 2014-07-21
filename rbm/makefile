SRC_FILES=Rbm.cpp main.cpp
HEADER_FILES=Rbm.h consts.h
CC=icc
CXX=icpc
FLAGS=-g -vec-report -mkl -O2 -fopenmp -openmp-report=2
all: rbm rbm.mic
rbm: ${SRC_FILES} ${HEADER_FILES} makefile
	${CXX} ${SRC_FILES} ${FLAGS} -o rbm
rbm.mic: ${SRC_FILES} ${HEADER_FILES} makefile
	${CXX} ${SRC_FILES} ${FLAGS} -mmic -o rbm.mic
clean:
	rm -f rbm
	rm -f rbm.mic
