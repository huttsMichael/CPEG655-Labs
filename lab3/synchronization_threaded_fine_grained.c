#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <papi.h>

int events[1] = {PAPI_TOT_CYC};
long long values[1];
int eventset;
int nEvents, retval;
char eventLabel[PAPI_MAX_STR_LEN];
const int N = 1048576; // 64, 1048576

struct p {
    int v;
    struct p* left;
    struct p* right;
    pthread_mutex_t node_lock; // Lock for each node
};

struct p* add(int v, struct p* somewhere) {
    if (somewhere == NULL) {
        struct p* newNode = (struct p*)malloc(sizeof(struct p));
        newNode->v = v;
        newNode->left = NULL;
        newNode->right = NULL;
        pthread_mutex_init(&newNode->node_lock, NULL);
        return newNode;
    }

    pthread_mutex_lock(&somewhere->node_lock);
    if (v < somewhere->v) {
        somewhere->left = add(v, somewhere->left);
    } else if (v > somewhere->v) {
        somewhere->right = add(v, somewhere->right);
    }
    pthread_mutex_unlock(&somewhere->node_lock);

    return somewhere;
}

struct p* delete(int v, struct p* somewhere) {
    if (somewhere == NULL) {
        return NULL; // key not found
    }

    pthread_mutex_lock(&somewhere->node_lock);
    if (v < somewhere->v) {
        somewhere->left = delete(v, somewhere->left);
    } else if (v > somewhere->v) {
        somewhere->right = delete(v, somewhere->right);
    } else {
        // node with the key 'v' found
        if (somewhere->left == NULL) {
            struct p* temp = somewhere->right;
            pthread_mutex_unlock(&somewhere->node_lock);
            free(somewhere);
            return temp;
        } else if (somewhere->right == NULL) {
            struct p* temp = somewhere->left;
            pthread_mutex_unlock(&somewhere->node_lock);
            free(somewhere);
            return temp;
        }

        // node with two children, get the inorder successor (smallest in the right subtree)
        struct p* temp = somewhere->right;
        while (temp->left != NULL) {
            temp = temp->left;
        }

        somewhere->v = temp->v;
        somewhere->right = delete(temp->v, somewhere->right);
    }
    pthread_mutex_unlock(&somewhere->node_lock);

    return somewhere;
}

int size(struct p* somewhere) {
    if (somewhere == NULL) {
        return 0;
    } else {
        int left_size, right_size;
        pthread_mutex_lock(&somewhere->node_lock);
        left_size = size(somewhere->left);
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
    right_integrity = checkIntegrity(somewhere->right);
    pthread_mutex_unlock(&somewhere->node_lock);

    if (somewhere->left != NULL && somewhere->left->v > somewhere->v) {
        return 0; // left subtree violates the property
    }

    if (somewhere->right != NULL && somewhere->right->v < somewhere->v) {
        return 0; // right subtree violates the property
    }

    return left_integrity && right_integrity;
}

void* workload() {
    pthread_t thread_id = pthread_self();
    // printf("Thread ID: %lu\n", thread_id);
    struct p* root = NULL;

    // add random keys to the tree
    for (int i = 0; i < 1000; i++) {
        int key = rand() % N + 1;
        root = add(key, root);
    }

    // add and remove random keys from the tree
    for (int i = 0; i < 100000; i++) {
        int key = rand() % N + 1;
        root = add(key, root);
        root = delete(key, root);
    }

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

    // Actual work goes here.
    for (int j = 0; j < 16; j++) {
        // printf("Creating thread: %i\n", j);
        pthread_create(&threads[j], NULL, workload, NULL);
        // printf("Done creating thread: %i\n", j);
    }

    for (int j = 0; j < 16; j++) {
        // printf("Joining thread: %i\n", j);
        pthread_join(threads[j], NULL);
        // printf("Done joining thread: %i\n", j);
    }

    if ((retval = PAPI_stop(eventset, values)) != PAPI_OK) {
        fprintf(stderr, "PAPI failed to read counters: %s\n", PAPI_strerror(retval));
        exit(1);
    }

    /* Print out your profiling results here */
    for (int i = 0; i < nEvents; i++) {
        PAPI_event_code_to_name(events[i], eventLabel);
        printf("%s:\t%lld\t", eventLabel, values[i]);
    }
    printf("\n");

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
