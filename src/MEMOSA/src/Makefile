.PHONY : all clean

CXX = h5c++

all: xyz2xdmf

xyz2xdmf:	xyz2xdmf.o
	$(CXX) -shlib -o $@ $^

clean:
	$(RM) *.o *~ xyz2xdmf

