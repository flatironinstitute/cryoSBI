#ifndef CONFIG_READER
#define CONFIG_READER

#include "definitions.h"
#include <thread>
#include <typeinfo>

void parse_args(int, char **, int, int, std::string &, int &);
void parse_input_file(std::string, myparam_t *, int);

#endif