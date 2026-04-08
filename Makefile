TARGET = $(patsubst %.cu, %, $(wildcard *.cu))

default: $(TARGET)

%: %.cu
	nvcc -O3 -g -o $@ $< -I/usr/include -L/usr/lib/x86_64-linux-gnu -lcuda

clean:
	rm -f $(TARGET)
