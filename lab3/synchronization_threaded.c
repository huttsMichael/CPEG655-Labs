#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <papi.h>

// define a mutex to protect the tree
pthread_mutex_t tree_mutex = PTHREAD_MUTEX_INITIALIZER;

// papi setup pulled from lab 2
int events[1] = {PAPI_TOT_CYC}; /*PAPI_L1_DCM, PAPI_L2_DCM, PAPI_TLB_DM*/
long long values[1];
int eventset;
int nEvents, retval;
char eventLabel[PAPI_MAX_STR_LEN];

// misc constants
const int N = 64; // 64, 1048576

struct p {
    int v;
    struct p * left;
    struct p * right;
};

struct p *add(int v, struct p *somewhere) {
    // printf("add\n");
    if (somewhere == NULL) {
        struct p *newNode = (struct p *)malloc(sizeof(struct p));
        newNode->v = v;
        newNode->left = NULL;
        newNode->right = NULL;
        return newNode;
    }

    if (v < somewhere->v) {
        somewhere->left = add(v, somewhere->left);
    } 
    else if (v > somewhere->v) {
        somewhere->right = add(v, somewhere->right);
    }

    return somewhere;
}

struct p *delete(int v, struct p *somewhere) {
    // printf("delete\n");
    if (somewhere == NULL) {
        return NULL; // key not found
    }

    if (v < somewhere->v) {
        somewhere->left = delete(v, somewhere->left);
    } 
    else if (v > somewhere->v) {
        somewhere->right = delete(v, somewhere->right);
    } 
    else {
        // node with the key 'v' found
        if (somewhere->left == NULL) {
            struct p *temp = somewhere->right;
            free(somewhere);
            return temp;
        } 
        else if (somewhere->right == NULL) {
            struct p *temp = somewhere->left;
            free(somewhere);
            return temp;
        }

        struct p *temp = somewhere->right;
        while (temp->left != NULL) {
            temp = temp->left;
        }

        somewhere->v = temp->v;
        somewhere->right = delete(temp->v, somewhere->right);
    }

    return somewhere;
}

int size(struct p *somewhere) {
    // printf("size\n");
    if (somewhere == NULL) {
        return 0;
    } 
    else {
        return 1 + size(somewhere->left) + size(somewhere->right);
    }
}

int checkIntegrity(struct p *somewhere) {
    // printf("checkIntegrity\n");
    if (somewhere == NULL) {
        return 1;
    }

    if (somewhere->left != NULL && somewhere->left->v > somewhere->v) {
        return 0; // left subtree violates the property
    }

    if (somewhere->right != NULL && somewhere->right->v < somewhere->v) {
        return 0; // right subtree violates the property
    }

    return checkIntegrity(somewhere->left) && checkIntegrity(somewhere->right);
}

void* workload() {
    // printf("workload\n");
    struct p *root = NULL;
    
    // add random keys to the tree
    for (int i = 0; i < 1000; i++) {
        int key = rand() % N + 1;
        pthread_mutex_lock(&tree_mutex);
        root = add(key, root);
        pthread_mutex_unlock(&tree_mutex);
    }
    
    // add and remove random keys from the tree
    for (int i = 0; i < 100000; i++) {
        int key = rand() % N + 1;
        pthread_mutex_lock(&tree_mutex);
        root = add(key, root);
        // pthread_mutex_unlock(&tree_mutex);
        // pthread_mutex_lock(&tree_mutex);
        root = delete(key, root);
        pthread_mutex_unlock(&tree_mutex);
    }

    // print size and checkIntegrity (not done when profiling)
    // pthread_mutex_lock(&tree_mutex);
    // printf("Size: %d\n", size(root));
    // printf("Tree integrity: %s\n", checkIntegrity(root) ? "Valid" : "Invalid");
    // pthread_mutex_unlock(&tree_mutex);

    pthread_exit(NULL);
}

int main() {
    srand(time(NULL));
    pthread_t threads[16];

    if (PAPI_VER_CURRENT != PAPI_library_init(PAPI_VER_CURRENT)) {
        printf("Can't initiate PAPI library!\n");
        exit(-1);
    }

    eventset = PAPI_NULL;
    if (PAPI_create_eventset(&eventset) != PAPI_OK) {
        printf("Can't create eventset!\n");
        exit(-3);
    }
    nEvents = sizeof(values) / sizeof(values[0]);
    for (int i = 0; i < nEvents; i++) {
        if ((retval = PAPI_add_event(eventset, events[i])) != PAPI_OK) {
        printf("\n\t   Error : PAPI failed to add event %d\n", i);
        printf("\n\t   Error string : %s  :: Error code : %d \n",
                PAPI_strerror(retval), retval);
        }
    }

    if ((retval = PAPI_start(eventset)) != PAPI_OK) {
        fprintf(stderr, "PAPI failed to start counters: %s\n",
                PAPI_strerror(retval));
        exit(1);
    }

    clock_t tic = clock();
    // printf("before workload\n");
    // Actual work goes here.
    for (int j = 0; j < 16; j++) {
        pthread_create(&threads[j], NULL, workload, NULL);
    }
    
    for (int j = 0; j < 16; j++) {
        pthread_join(threads[j], NULL);
    }
    // printf("after workload\n");
    clock_t toc = clock();

    printf("%f\n", (double)(toc - tic) / CLOCKS_PER_SEC);

    if ((retval = PAPI_stop(eventset, values)) != PAPI_OK) {
        fprintf(stderr, "PAPI failed to read counters: %s\n",
                PAPI_strerror(retval));
        exit(1);
    }

      /* Print out your profiling results here */
    // for (int i = 0; i < nEvents; i++) {
    //     PAPI_event_code_to_name(events[i], eventLabel);
    //     printf("%s:\t%lld\t", eventLabel, values[i]);
    // }
    // printf("\n");

    if ((retval = PAPI_cleanup_eventset(eventset)) != PAPI_OK) {
        printf(
            "\n\t   Error : PAPI failed to clean the events from created Eventset");
        printf("\n\t   Error string : %s  :: Error code : %d \n",
            PAPI_strerror(retval), retval);
        return (-1);
    }
    if ((retval = PAPI_destroy_eventset(&eventset)) != PAPI_OK) {
        printf(
            "\n\t   Error : PAPI failed to clean the events from created Eventset");
        printf("\n\t   Error string : %s  :: Error code : %d \n",
            PAPI_strerror(retval), retval);
        return (-1);
    }
    PAPI_shutdown();

    return 0;
}