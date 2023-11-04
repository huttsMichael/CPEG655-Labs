#include <stdio.h>
#include <stdlib.h>

struct p {
    int v;
    struct p * left;
    struct p * right;
};

struct p *add(int v, struct p *somewhere) {
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
    if (somewhere == NULL) {
        return NULL; 
    }

    if (v < somewhere->v) {
        somewhere->left = delete(v, somewhere->left);
    } 
    else if (v > somewhere->v) {
        somewhere->right = delete(v, somewhere->right);
    } 
    else {
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
    if (somewhere == NULL) {
        return 0;
    } 
    else {
        return 1 + size(somewhere->left) + size(somewhere->right);
    }
}

int checkIntegrity(struct p *somewhere) {
    if (somewhere == NULL) {
        return 1;
    }

    if (somewhere->left != NULL && somewhere->left->v > somewhere->v) {
        return 0; 
    }

    if (somewhere->right != NULL && somewhere->right->v < somewhere->v) {
        return 0; 
    }

    return checkIntegrity(somewhere->left) && checkIntegrity(somewhere->right);
}

int main() {
    struct p *root = NULL;
    
    root = add(7, root);
    root = add(5, root);
    root = add(9, root);
    root = add(4, root);
    root = add(6, root);
    root = add(8, root);
    root = add(10, root);
    root = add(2, root);
    root = add(12, root);
    
    printf("Size of tree: %d\n", size(root));
    printf("Tree integrity: %s\n", checkIntegrity(root) ? "Valid" : "Invalid");
    
    root = delete(5, root);
    root = delete(9, root);
    
    printf("Size of tree after deletion: %d\n", size(root));
    printf("Tree integrity after deletion: %s\n", checkIntegrity(root) ? "Valid" : "Invalid");

    return 0;
}
