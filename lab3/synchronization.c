struct p {
    int v;
    struct p * left;
    struct p * right;
};

struct p * add (int v, struct p * somewhere);
struct p * remove(int v, struct p *somewhere);
int size();
int checkIntegrity();