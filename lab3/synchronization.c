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
    } else if (v > somewhere->v) {
        somewhere->right = add(v, somewhere->right);
    }

    return somewhere;
}

struct p *delete(int v, struct p *somewhere) {
    if (somewhere == NULL) {
        return NULL; // Key not found
    }

    if (v < somewhere->v) {
        somewhere->left = delete(v, somewhere->left);
    } else if (v > somewhere->v) {
        somewhere->right = delete(v, somewhere->right);
    } else {
        // Node with the key 'v' found
        if (somewhere->left == NULL) {
            struct p *temp = somewhere->right;
            free(somewhere);
            return temp;
        } else if (somewhere->right == NULL) {
            struct p *temp = somewhere->left;
            free(somewhere);
            return temp;
        }

        // Node with two children, get the inorder successor (smallest in the right subtree)
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
    } else {
        return 1 + size(somewhere->left) + size(somewhere->right);
    }
}

int checkIntegrity(struct p *somewhere) {
    if (somewhere == NULL) {
        return 1;
    }

    if (somewhere->left != NULL && somewhere->left->v > somewhere->v) {
        return 0; // Left subtree violates the property
    }

    if (somewhere->right != NULL && somewhere->right->v < somewhere->v) {
        return 0; // Right subtree violates the property
    }

    return checkIntegrity(somewhere->left) && checkIntegrity(somewhere->right);
}

int main() {
    struct p *root = NULL;
    
    root = add(50, root);
    root = add(30, root);
    root = add(70, root);
    root = add(20, root);
    root = add(40, root);
    root = add(60, root);
    root = add(80, root);
    
    printf("Size of tree: %d\n", size(root));
    printf("Tree integrity: %s\n", checkIntegrity(root) ? "Valid" : "Invalid");
    
    root = delete(30, root);
    root = delete(70, root);
    
    printf("Size of tree after deletion: %d\n", size(root));
    printf("Tree integrity after deletion: %s\n", checkIntegrity(root) ? "Valid" : "Invalid");

    return 0;
}
