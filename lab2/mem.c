#include <papi.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned len = 16*(1 << 20);

// Problem 1.1
/*
typedef struct Mem {
	int fc;
	int fe;
	int fa;
	char fb;
	char fd;
} Mem;
*/

// Problem 1.2
typedef struct MemCE {
	int fc;
	int fe;
} MemCE;

typedef struct MemABD {
	int fa;
	char fb;
	char fd;
} MemABD;


// Problem 1.1
/*
void func(Mem * a)
{
	int i;

	for(i=0; i<len; i++){
		a[i].fa = a[i].fb+a[i].fd;
	}

	for(i=0; i<len; i++){
		a[i].fc = a[i].fe*2;
	}

}
*/


// Problem 1.2
void func(MemCE * ce, MemABD * abd)
{
	int i;

	for(i=0; i<len; i++){
		abd[i].fa = abd[i].fb+abd[i].fd;
	}

	for(i=0; i<len; i++){
		ce[i].fc = ce[i].fe*2;
	}

}

/* Please add your events here */
int events[1] = {PAPI_L2_DCM}; /*PAPI_L1_DCM, PAPI_L2_DCM, PAPI_TLB_DM*/
int eventnum = 1;

int main()
{
	long long values[1];
	int eventset;
	// Problem 1.1
	/*
	Mem * a;
	*/

	// Problem 1.2
	MemCE * ce;
	MemABD * abd;

	if(PAPI_VER_CURRENT != PAPI_library_init(PAPI_VER_CURRENT)){
		printf("Can't initiate PAPI library!\n");
		exit(-1);
	}

	eventset = PAPI_NULL;
	if(PAPI_create_eventset(&eventset) != PAPI_OK){
		printf("Can't create eventset!\n");
		exit(-3);
	}
	if(PAPI_OK != PAPI_add_events(eventset, events, eventnum)){
		printf("Can't add events!\n");
		exit(-4);
	}

	// Problem 1.1
	/*
	a = (Mem *) malloc(len*sizeof(Mem));
	*/

	// Problem 1.2
	ce = (MemCE *) malloc(len*sizeof(MemCE));
	abd = (MemABD *) malloc(len*sizeof(MemABD));
	PAPI_start(eventset);
	// Problem 1.1
	/*
	func(a);
	*/

	// Problem 1.2
	func(ce, abd);
	PAPI_stop(eventset, values);
	// Problem 1.1
	/*
	free(a);
	*/

	// Problem 1.2
	free(ce);
	free(abd);

	/*Print out PAPI reading*/
	char event_name[PAPI_MAX_STR_LEN];
	if (PAPI_event_code_to_name( events[0], event_name ) == PAPI_OK)
		printf("%s: %lld\n", event_name, values[0]);
	
	return EXIT_SUCCESS;
}
