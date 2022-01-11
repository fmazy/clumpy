# -fpic means Position Independent Code. It is necessary when compiling shared libraries
# -shared is used for creating a shared library
libashfunc.so: _ash_func.cpp
	gcc -fpic -Wall -shared _ash_func.cpp -o ~/anaconda3/lib/libashfunc.so
