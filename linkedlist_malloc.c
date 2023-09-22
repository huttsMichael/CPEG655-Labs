#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>

#include "papi.h"

#define RDTSC(a) __asm__ __volatile__ ("rdtsc" : "=a" (a.u32lo), "=d" (a.u32hi))

#define fieldarraysize (8)

typedef struct List {
    int key;

    struct List * next;
} List;


void create_list_1(List ** head, int size)
{
    if(0 == size){
        return;
    }

    List ** cur_head = head;

    for(int i = 0; i < size; ++i){
        *cur_head = (List *) malloc(sizeof(List));

        cur_head = &((*cur_head)->next);
    }

    (*cur_head) = NULL;
}

void create_list_2(List ** head, int size, int block_n)
{
    if(0 == size){
        return;
    }

    List ** cur_head = head;

    for(int i = 0; i < size; i+=block_n){
        while(*cur_head) {
            cur_head = & ((*cur_head)->next);
        }

        *cur_head = (List *) malloc(sizeof(List)*block_n);

        List * p_tmp = *cur_head;
        for(int j = 0; j < block_n-1; ++j){
            p_tmp->next = p_tmp + 1;

            p_tmp = p_tmp->next;
        }

        p_tmp->next = NULL;
    }
}

void flush(int * g, int size){
	int i;

	for(i = 0; i<size; i++){
		g[i]++;
	}
}

int main(){
	int * garbage;
	
    List * list_head1 = NULL;
    List * list_head2 = NULL;

	const int arraysize = 8*(1 << 20);
	int i, j, stride;
	int iter, numOfIter;
	struct timeval start, end;
	int r;

	int nEvents, retval;
    int EventSet = PAPI_NULL;
    int events[] = {PAPI_L2_DCM}; //, PAPI_L2_TCM};
    long_long values[] = {0, 0};
    char eventLabel[PAPI_MAX_STR_LEN];

    printf("sizeof(List) = %d \n", sizeof(List));

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
        	PAPI_event_code_to_name(events[i], eventLabel);
			printf("\n\t   Error : PAPI failed to add event %s\n", eventLabel);
			printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
		}
	}

	srand(time(NULL));

	garbage = (int *) malloc(sizeof(int) * arraysize);
	for (i = 0; i < arraysize; i++) {
		garbage[i] = rand();
	}

   
    create_list_1(&list_head1, arraysize);
    create_list_2(&list_head2, arraysize, 4);

    printf("list1: %x, %x\n", list_head1, list_head1->next);
    printf("list2: %x, %x\n", list_head2, list_head2->next);

	numOfIter = 10;

	/* ver. 1: s1 */

	if ((retval = PAPI_start(EventSet)) != PAPI_OK) {
        fprintf(stderr, "PAPI failed to start counters: %s\n", PAPI_strerror(retval));
        exit(1);
    }

	for(iter = 0; iter < numOfIter; iter++){
        List * p_tmp = list_head1;
        while(p_tmp) {
            (p_tmp->key)++;

            p_tmp = p_tmp->next;
        }
	}

	if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK) {
        fprintf(stderr, "PAPI failed to read counters: %s\n", PAPI_strerror(retval));
        exit(1);
    }

	printf("ver. 1:\n");

	for(i = 0; i < nEvents; i++){
        PAPI_event_code_to_name(events[i], eventLabel);
        printf("%s:\t%lld\t", eventLabel, values[i]);
    }
    printf("\n");


	/* ver. 2: s2 */

	if ((retval = PAPI_start(EventSet)) != PAPI_OK) {
        fprintf(stderr, "PAPI failed to start counters: %s\n", PAPI_strerror(retval));
        exit(1);
    }


	for(iter = 0; iter < numOfIter; iter++){
        List * p_tmp = list_head2;
        while(p_tmp) {
            (p_tmp->key)++;

            p_tmp = p_tmp->next;
        }
	}

	if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK) {
        fprintf(stderr, "PAPI failed to read counters: %s\n", PAPI_strerror(retval));
        exit(1);
    }

	printf("ver. 2:\n");

	for(i = 0; i < nEvents; i++){
        PAPI_event_code_to_name(events[i], eventLabel);
        printf("%s:\t%lld\t", eventLabel, values[i]);
    }
    printf("\n");

	/* ver. 2: s2 */

	printf("list1 = %d\n", list_head1->key);
	printf("list2 = %d\n", list_head2->key);

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
