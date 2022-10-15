CC = gcc
CFLAGS = -Wall -O3 -lm -g -std=c99

example:
	@$(CC) -o example example.c $(CFLAGS)

.PHONY: clean
clean:
	@rm -f example test test.c

.PHONY: test
test: test.c
	@$(CC) -o test test.c $(CFLAGS) && ./test

test.c:
	@echo "#define LIBDTREE_TEST_\n#include \"libdtree.h\"\nint main(){ run_tests(); }" > test.c
