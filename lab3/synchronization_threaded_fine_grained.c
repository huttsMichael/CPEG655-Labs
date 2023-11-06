#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <papi.h>

pthread_mutex_t initialLock = PTHREAD_MUTEX_INITIALIZER;
int events[1] = {PAPI_TOT_CYC};
long long values[1];
int eventset;
int nEvents, retval;
char eventLabel[PAPI_MAX_STR_LEN];
const int N = 1048576; // 64, 1048576
struct p* root = NULL;

struct p {
    int v;
    struct p* left;
    struct p* right;
    pthread_mutex_t node_lock;
};

struct p* add(int v, struct p* somewhere) {
    struct p* currentNode = somewhere;
    struct p* parentNode = NULL;
    
    while (currentNode != NULL) {
        pthread_mutex_lock(&currentNode->node_lock);
        
        if (parentNode != NULL) {
            pthread_mutex_unlock(&parentNode->node_lock);
        }
        
        parentNode = currentNode;
        
        if (v < currentNode->v) {
            if (currentNode->left == NULL) {
                struct p* newNode = (struct p *)malloc(sizeof(struct p));
                newNode->v = v;
                newNode->left = NULL;
                newNode->right = NULL;
                pthread_mutex_init(&newNode->node_lock, NULL);
                currentNode->left = newNode; 
                pthread_mutex_unlock(&currentNode->node_lock);
                return somewhere;
            } else {
                currentNode = currentNode->left;
            }
        } else if (v > currentNode->v) {
            if (currentNode->right == NULL) {
                struct p* newNode = (struct p *)malloc(sizeof(struct p));
                newNode->v = v;
                newNode->left = NULL;
                newNode->right = NULL;
                pthread_mutex_init(&newNode->node_lock, NULL);
                currentNode->right = newNode; 
                pthread_mutex_unlock(&currentNode->node_lock);
                return somewhere;
            } else {
                currentNode = currentNode->right;
            }
        } else {
            // not explicitly specified, but default to right for duplicate values
            if (currentNode->right == NULL) {
                struct p* newNode = (struct p *)malloc(sizeof(struct p));
                newNode->v = v;
                newNode->left = NULL;
                newNode->right = NULL;
                pthread_mutex_init(&newNode->node_lock, NULL);
                currentNode->right = newNode; 
                pthread_mutex_unlock(&currentNode->node_lock);
                return somewhere;
            } else {
                currentNode = currentNode->right;
            }
        }
    }
    
    // Handle the case where the tree is initially empty
    struct p* newNode = (struct p *)malloc(sizeof(struct p));
    newNode->v = v;
    newNode->left = NULL;
    newNode->right = NULL;
    pthread_mutex_init(&newNode->node_lock, NULL);
    return newNode;
}

struct p* delete(int v, struct p* somewhere) {
    // printf("delete\n");
    if (somewhere == NULL) {
        return NULL; // key not found
    }

    pthread_mutex_lock(&somewhere->node_lock);
    if (v < somewhere->v) {
        somewhere->left = delete(v, somewhere->left);
        pthread_mutex_unlock(&somewhere->node_lock); 
    } 
    else if (v > somewhere->v) {
        somewhere->right = delete(v, somewhere->right);
        pthread_mutex_unlock(&somewhere->node_lock);
    } 
    else {
        // node with the key 'v' found
        if (somewhere->left == NULL) {
            struct p* temp = somewhere->right;
            // pthread_mutex_unlock(&somewhere->node_lock); 
            free(somewhere);
            return temp;
        } 
        else if (somewhere->right == NULL) {
            struct p* temp = somewhere->left;
            // pthread_mutex_unlock(&somewhere->node_lock); 
            free(somewhere);
            return temp;
        }

        struct p* temp = somewhere->right;
        while (temp->left != NULL) {
            temp = temp->left;
        }

        somewhere->v = temp->v;
        somewhere->right = delete(temp->v, somewhere->right);
        pthread_mutex_unlock(&somewhere->node_lock); 
    }

    return somewhere;
}

int size(struct p* somewhere) {
    if (somewhere == NULL) {
        return 0;
    } 
    else {
        int left_size, right_size;

        pthread_mutex_lock(&somewhere->node_lock);
        left_size = size(somewhere->left);
        pthread_mutex_unlock(&somewhere->node_lock);

        pthread_mutex_lock(&somewhere->node_lock);
        right_size = size(somewhere->right);
        pthread_mutex_unlock(&somewhere->node_lock);

        return 1 + left_size + right_size;
    }
}

int checkIntegrity(struct p* somewhere) {
    if (somewhere == NULL) {
        return 1;
    }

    int left_integrity, right_integrity;

    pthread_mutex_lock(&somewhere->node_lock);
    left_integrity = checkIntegrity(somewhere->left);
    pthread_mutex_unlock(&somewhere->node_lock); 

    pthread_mutex_lock(&somewhere->node_lock);
    right_integrity = checkIntegrity(somewhere->right);
    pthread_mutex_unlock(&somewhere->node_lock); 

    // left failure condition
    if (somewhere->left != NULL && somewhere->left->v > somewhere->v) {
        return 0;
    }

    // right failure condition
    if (somewhere->right != NULL && somewhere->right->v < somewhere->v) {
        return 0;
    }

    return left_integrity && right_integrity;
}

void* workload() {
    // pthread_t thread_id = pthread_self();
    // printf("Thread ID: %lu\n", thread_id);
    // struct p* root = NULL;


    // add random keys to the tree
    for (int i = 0; i < 1000; i++) {
        int key = rand() % N + 1;
        root = add(key, root);
        // printf("Size (%lu): %d\n", thread_id, size(root));
        // printf("Tree integrity: %s\n", checkIntegrity(root) ? "Valid" : "Invalid");
    }

    // add and remove random keys from the tree
    for (int i = 0; i < 100000; i++) {
        int key = rand() % N + 1;
        root = add(key, root);
        root = delete(key, root);
    }

    // print size and checkIntegrity (not done when profiling)
    // printf("Size: %d\n", size(root));
    // printf("Tree integrity: %s\n", checkIntegrity(root) ? "Valid" : "Invalid");

    // printf("Exiting Thread ID: %lu\n", thread_id);
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
            printf("\n\t   Error string : %s  :: Error code : %d \n", PAPI_strerror(retval), retval);
        }
    }

    if ((retval = PAPI_start(eventset)) != PAPI_OK) {
        fprintf(stderr, "PAPI failed to start counters: %s\n",
                PAPI_strerror(retval));
        exit(1);
    }

    // initialize tree to prevent weirdness
    // root = add(N, root);

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
        printf("\n\t   Error : PAPI failed to clean the events from created Eventset");
        printf("\n\t   Error string : %s  :: Error code : %d \n", PAPI_strerror(retval), retval);
        return (-1);
    }
    if ((retval = PAPI_destroy_eventset(&eventset)) != PAPI_OK) {
        printf("\n\t   Error : PAPI failed to clean the events from created Eventset");
        printf("\n\t   Error string : %s  :: Error code : %d \n", PAPI_strerror(retval), retval);
        return (-1);
    }
    PAPI_shutdown();

    return 0;
}
