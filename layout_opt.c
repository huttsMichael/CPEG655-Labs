#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "papi.h"

#define RDTSC(a) __asm__ __volatile__("rdtsc" : "=a"(a.u32lo), "=d"(a.u32hi))

#define fieldarraysize (8)

typedef struct s1 {
  int a[fieldarraysize];
  int b[fieldarraysize];
  int c[fieldarraysize];
  int d[fieldarraysize];
} s1;

typedef struct s2 {
  int a[fieldarraysize];
  int d[fieldarraysize];
  int c[fieldarraysize];
  int b[fieldarraysize];
} s2;

#define arraysize ((1 << 23))

typedef struct s3 {
  struct {
    int a[fieldarraysize];
    int d[fieldarraysize];
  } ad[arraysize];

  struct {
    int c[fieldarraysize];
    int b[fieldarraysize];
  } cb[arraysize];
} s3;

void flush(int *g, int size) {
  int i;

  for (i = 0; i < size; i++) {
    g[i]++;
  }
}

int main() {
  int *garbage;

  s1 *a1;
  s2 *a2;
  s3 *a3;

  int i, j, stride;
  int iter, numOfIter;
  struct timeval start, end;
  int r;

  int nEvents, retval;
  int EventSet = PAPI_NULL;
  int events[] = {PAPI_LST_INS, PAPI_TLB_DM}; //, PAPI_L2_TCM};
  long_long values[] = {0, 0, 0, 0};
  char eventLabel[PAPI_MAX_STR_LEN];

  if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
    printf("\n\t  Error : PAPI Library initialization error! \n");
    return (-1);
  }

  if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK) {
    printf("\n\t  Error : PAPI failed to create the Eventset\n");
    printf("\n\t  Error string : %s  :: Error code : %d \n",
           PAPI_strerror(retval), retval);
    return (-1);
  }

  nEvents = sizeof(events) / sizeof(events[0]);
  for (i = 0; i < nEvents; i++) {
    if ((retval = PAPI_add_event(EventSet, events[i])) != PAPI_OK) {
      PAPI_event_code_to_name(events[i], eventLabel);
      printf("\n\t   Error : PAPI failed to add event %s\n", eventLabel);
      printf("\n\t   Error string : %s  :: Error code : %d \n",
             PAPI_strerror(retval), retval);
    }
  }

  srand(time(NULL));

  garbage = (int *)malloc(sizeof(int) * arraysize);
  for (i = 0; i < arraysize; i++) {
    garbage[i] = rand();
  }

  a1 = (s1 *)malloc(sizeof(s1) * arraysize);
  a2 = (s2 *)malloc(sizeof(s2) * arraysize);
  a3 = (s3 *)malloc(sizeof(s3));

  numOfIter = 10;

  /* ver. 1: s1 */

  if ((retval = PAPI_start(EventSet)) != PAPI_OK) {
    fprintf(stderr, "PAPI failed to start counters: %s\n",
            PAPI_strerror(retval));
    exit(1);
  }

  for (iter = 0; iter < numOfIter; iter++) {
    for (i = 0; i < arraysize; i++) {
      for (j = 0; j < fieldarraysize; j++) {
        a1[i].a[j] = a1[i].d[j] * 2;
      }

      for (j = 0; j < fieldarraysize; j++) {
        a1[i].b[j] = a1[i].c[j] * 2;
      }
    }
  }

  if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK) {
    fprintf(stderr, "PAPI failed to read counters: %s\n",
            PAPI_strerror(retval));
    exit(1);
  }

  printf("ver. 1:\n");

  for (i = 0; i < nEvents; i++) {
    PAPI_event_code_to_name(events[i], eventLabel);
    printf("%s:\t%lld\t", eventLabel, values[i]);
  }
  printf("\n");

  /* ver. 2: s2 */

  if ((retval = PAPI_start(EventSet)) != PAPI_OK) {
    fprintf(stderr, "PAPI failed to start counters: %s\n",
            PAPI_strerror(retval));
    exit(1);
  }

  for (iter = 0; iter < numOfIter; iter++) {
    for (i = 0; i < arraysize; i++) {
      for (j = 0; j < fieldarraysize; j++) {
        a2[i].a[j] = a2[i].d[j] * 2;
      }

      for (j = 0; j < fieldarraysize; j++) {
        a2[i].b[j] = a2[i].c[j] * 2;
      }
    }
  }

  if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK) {
    fprintf(stderr, "PAPI failed to read counters: %s\n",
            PAPI_strerror(retval));
    exit(1);
  }

  printf("ver. 2:\n");

  for (i = 0; i < nEvents; i++) {
    PAPI_event_code_to_name(events[i], eventLabel);
    printf("%s:\t%lld\t", eventLabel, values[i]);
  }
  printf("\n");

  /* ver. 2: s2 */

  /* ver. 3: s3 */

  if ((retval = PAPI_start(EventSet)) != PAPI_OK) {
    fprintf(stderr, "PAPI failed to start counters: %s\n",
            PAPI_strerror(retval));
    exit(1);
  }

  for (iter = 0; iter < numOfIter; iter++) {
    for (i = 0; i < arraysize; i++) {
      for (j = 0; j < fieldarraysize; j++) {
        a3->ad[i].a[j] = a3->ad[i].d[j] * 2;
      }

      for (j = 0; j < fieldarraysize; j++) {
        a3->cb[i].b[j] = a3->cb[i].c[j] * 2;
      }
    }
  }

  if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK) {
    fprintf(stderr, "PAPI failed to read counters: %s\n",
            PAPI_strerror(retval));
    exit(1);
  }

  printf("ver. 3:\n");

  for (i = 0; i < nEvents; i++) {
    PAPI_event_code_to_name(events[i], eventLabel);
    printf("%s:\t%lld\t", eventLabel, values[i]);
  }
  printf("\n");

  /* ver. 3: s3 */

  printf("a1 = %d\n", a1[rand() % arraysize].a[3]);
  printf("a2 = %d\n", a2[rand() % arraysize].b[3]);
  printf("a3 = %d\n", a3->ad[rand() % arraysize].a[3]);

  printf("garbage1 = %d\n", garbage[rand() % arraysize]);
  printf("garbage2 = %d\n", garbage[rand() % arraysize]);
  printf("garbage3 = %d\n", garbage[rand() % arraysize]);

  free(a1);
  free(a2);

  free(garbage);

  /* Gentally shutdown PAPI */
  if ((retval = PAPI_cleanup_eventset(EventSet)) != PAPI_OK) {
    printf(
        "\n\t   Error : PAPI failed to clean the events from created Eventset");
    printf("\n\t   Error string : %s  :: Error code : %d \n",
           PAPI_strerror(retval), retval);
    return (-1);
  }
  if ((retval = PAPI_destroy_eventset(&EventSet)) != PAPI_OK) {
    printf(
        "\n\t   Error : PAPI failed to clean the events from created Eventset");
    printf("\n\t   Error string : %s  :: Error code : %d \n",
           PAPI_strerror(retval), retval);
    return (-1);
  }
  PAPI_shutdown();
}
