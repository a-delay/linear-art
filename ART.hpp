#include <stdlib.h>    // malloc, free
#include <string.h>    // memset, memcpy
#include <stdint.h>    // integer types
#include <emmintrin.h> // x86 SSE intrinsics
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>  // gettime
#include <algorithm>   // std::random_shuffle

// Constants for the node types
static const int8_t NodeType4=0;
static const int8_t NodeType16=1;
static const int8_t NodeType48=2;
static const int8_t NodeType256=3;
static const int8_t NodeTypeLinear=4;

// The maximum prefix length for compressed paths stored in the
// header, if the path is longer it is loaded from the database on
// demand
static const unsigned maxPrefixLength=9;

static const int8_t NODE4_SIZE = 4;
static const int8_t NODE48_SIZE = 24;

static const int8_t COUNT_NODES = 0;
static const int8_t CHILDREN_NODES = 1;


// Shared header of all inner nodes
struct Node {
   // length of the compressed path (prefix)
   uint32_t prefixLength;
   // number of non-null children
   uint16_t count;
   // node type
   int8_t type;
   // compressed path (prefix)
   uint8_t prefix[maxPrefixLength];

   Node(int8_t type) : prefixLength(0),count(0),type(type) {}
};

// Node with up to 4 children
struct Node4 : Node {
   uint8_t key[NODE4_SIZE];
   Node* child[NODE4_SIZE];

   Node4() : Node(NodeType4) {
      memset(key,0,sizeof(key));
      memset(child,0,sizeof(child));
   }
};

// Node with up to 16 children
struct Node16 : Node {
   uint8_t key[16];
   Node* child[16];

   Node16() : Node(NodeType16) {
      memset(key,0,sizeof(key));
      memset(child,0,sizeof(child));
   }
};

static const int8_t LINEAR_SIZE=10;
struct NodeLinear : Node {
   Node* child[LINEAR_SIZE];
   double a=0.0, b=0.0;

   NodeLinear() : Node(NodeTypeLinear) {
      memset(child,0,sizeof(child));
   }
};



static const uint8_t emptyMarker=NODE48_SIZE;

// Node with up to 48 children
struct Node48 : Node {
   uint8_t childIndex[256];
   Node* child[NODE48_SIZE];

   Node48() : Node(NodeType48) {
      memset(childIndex,emptyMarker,sizeof(childIndex));
      memset(child,0,sizeof(child));
   }
};

// Node with up to 256 children
struct Node256 : Node {
   Node* child[256];

   Node256() : Node(NodeType256) {
      memset(child,0,sizeof(child));
   }
};

void insert(Node*, Node**, uint8_t*, unsigned, uintptr_t, unsigned);
void loadKey(uintptr_t, uint8_t*);
Node* lookup(Node*, uint8_t*, unsigned, unsigned, unsigned);
inline uintptr_t getLeafValue(Node* node) {
   // The the value stored in the pseudo-leaf
   return reinterpret_cast<uintptr_t>(node)>>1;
}

inline bool isLeaf(Node* node) {
   // Is the node a leaf?
   return reinterpret_cast<uintptr_t>(node)&1;
}