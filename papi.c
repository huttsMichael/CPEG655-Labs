#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>

#include <papi.h>

typedef union
{
	unsigned long long u64;
	struct
	{
		unsigned long u32lo;
		unsigned long u32hi;
	};
	double d;
} Big;


#define RDTSC(a) __asm__ __volatile__ ("rdtsc" : "=a" (a.u32lo), "=d" (a.u32hi))

void flush(int * g, int size){
	int i;

	for(i = 0; i<size; i++){
		g[i]++;
	}
}

int main(){
	int * a;
	int * garbage;

	const int arraysize = 4*(1 << 20);
	int i, j, stride;
	int iter, numOfIter;
	struct timeval start, end;
	int r;

	Big bstart, bend;

	int nEvents, retval;
    int EventSet = PAPI_NULL;
    int events[] = {PAPI_TOT_CYC, PAPI_L2_TCM};
    long_long values[] = {0, 0};
    char eventLabel[PAPI_MAX_STR_LEN];

	if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
    {
        printf("\n\t  Error : PAPI Library initialization error! \n");
        return(-1);
    }

    if((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
    {   printf("\n\t  Error : PAPI failed to create the Eventset\n");
        printf("\n\t  Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
        return(-1);
    }

    nEvents = sizeof(events)/sizeof(events[0]);
    for(i = 0; i < nEvents; i++){
        if((retval = PAPI_add_event(EventSet, events[i])) != PAPI_OK)
        {
            printf("\n\t   Error : PAPI failed to add event %d\n", i);
            printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
        }
    }

	srand(time(NULL));

	garbage = (int *) malloc(sizeof(int) * arraysize);
	for (i = 0; i < arraysize; i++) {
		garbage[i] = rand();
	}

	a = (int *) malloc(sizeof(int) * arraysize);
	for (i = 0; i < arraysize; i++) {
		a[i] = rand();
	}

  printf("a[0] address = %x\n", &(a[0]));

	numOfIter = 10;

	/* ver. 1: gettimeofday */

	flush(garbage, arraysize);

	gettimeofday(&start, NULL);

	for(iter = 0; iter < numOfIter; iter++){
		for (i = 0; i < arraysize; i++) {
			a[i] = a[i] * 2;
		}
	}

	gettimeofday(&end, NULL);

	printf("v1: %lf us\n", (((end.tv_sec * 1000000 + end.tv_usec)
					- (start.tv_sec * 1000000 + start.tv_usec)))*1.0/numOfIter);

	/* ver. 2: tsc*/

	flush(garbage, arraysize);

	RDTSC(bstart);

	for(iter = 0; iter < numOfIter; iter++){
		for (i = 0; i < arraysize; i++) {
			a[i] = a[i] * 2;
		}
	}

	RDTSC(bend);

	printf("v2: %lld cycles\n", (bend.u64-bstart.u64));

	/* ver. 3: papi*/
	flush(garbage, arraysize);

	if ((retval = PAPI_start(EventSet)) != PAPI_OK) {
        fprintf(stderr, "PAPI failed to start counters: %s\n", PAPI_strerror(retval));
        exit(1);
    }


	for(iter = 0; iter < numOfIter; iter++){
		for (i = 0; i < arraysize; i++) {
			a[i] = a[i] * 2;
		}
	}

	if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK) {
        fprintf(stderr, "PAPI failed to read counters: %s\n", PAPI_strerror(retval));
        exit(1);
    }

	printf("ver. 3:\n");

	for(i = 0; i < nEvents; i++){
        PAPI_event_code_to_name(events[i], eventLabel);
        printf("%s:\t%lld\t", eventLabel, values[i]);
    }
    printf("\n");

	/* ver. 4: strided with papi*/
	flush(garbage, arraysize);

	if ((retval = PAPI_start(EventSet)) != PAPI_OK) {
        fprintf(stderr, "PAPI failed to start counters: %s\n", PAPI_strerror(retval));
        exit(1);
    }


	stride = 1024;
	for(iter = 0; iter < numOfIter; iter++){
		for (i = 0; i < stride; i++) {
			for(j = 0; j < arraysize; j+=stride){
				a[j] = a[j] * 2;
			}
		}
	}

    if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK) {
        fprintf(stderr, "PAPI failed to read counters: %s\n", PAPI_strerror(retval));
        exit(1);
    }

	printf("ver. 4:\n");

	for(i = 0; i < nEvents; i++){
        PAPI_event_code_to_name(events[i], eventLabel);
        printf("%s:\t%lld\t", eventLabel, values[i]);
    }
    printf("\n");

	r = rand() % arraysize;
	printf("a[%d] = %d\n", r, a[r]);

	free(a);
	free(garbage);

	/* Gentally shutdown PAPI */
    if((retval = PAPI_cleanup_eventset(EventSet)) != PAPI_OK)
    {
        printf("\n\t   Error : PAPI failed to clean the events from created Eventset");
        printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
        return(-1);
    }
    if((retval = PAPI_destroy_eventset(&EventSet)) != PAPI_OK)
    {
        printf("\n\t   Error : PAPI failed to clean the events from created Eventset");
        printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
        return(-1);
    }
    PAPI_shutdown();

}
