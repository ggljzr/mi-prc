#include <stdio.h>

int main(int argc, char * argv[])
{
	int n_devices = 0;
	cudaGetDeviceCount(&n_devices);

	for(int i = 0; i < n_devices; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device number: %d\n", i);
		printf("Device name: %s\n", prop.name);
		printf("CC: %d.%d\n", prop.major, prop.minor);
	}

	return 0;
}