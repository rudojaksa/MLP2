PACKAGE := MLP2

CFLAGS := -O0
CFLAGS := -O3 -ffast-math # MAC
CFLAGS := -Ofast -march=native -fopt-info -s

MLP2: MLP2.c Makefile
	gcc -std=gnu99 $(CFLAGS) -o $@ $< -lm

clean:
	rm -fv MLP2

-include ~/.github/Makefile.git
