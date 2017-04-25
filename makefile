CC=gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS=-lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

SRC := src
BIN := bin
DEMO := demo_scripts
PY := py_scripts

objects = word2vec word2phrase distance word-analogy compute-accuracy

all: dir $(objects) scripts_exec

dir:
	mkdir -p $(BIN)

$(objects): %: $(SRC)/%.c
	$(CC) $< -o $(BIN)/$@ $(CFLAGS)

scripts_exec:
	chmod +x $(DEMO)/*.sh
	chmod +x $(PY)/*.py

clean:
	rm -rf $(BIN)
