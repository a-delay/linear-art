/*
  Adaptive Radix Tree
  Viktor Leis, 2012
  leis@in.tum.de
 */

#include <stdlib.h>    // malloc, free
#include <string.h>    // memset, memcpy
#include <stdint.h>    // integer types
#include <emmintrin.h> // x86 SSE intrinsics
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>  // gettime
#include <algorithm>   // std::random_shuffle
#include "ART.hpp"
#include <unicode/ucol.h>
#include <vector>

inline Node* makeLeaf(uintptr_t tid) {
   // Create a pseudo-leaf
   return reinterpret_cast<Node*>((tid<<1)|1);
}



uint8_t flipSign(uint8_t keyByte) {
   // Flip the sign bit, enables signed SSE comparison of unsigned values, used by Node16
   return keyByte^128;
}

void loadKey(uintptr_t tid,uint8_t key[]) {
   // Store the key of the tuple into the key vector
   // Implementation is database specific
   reinterpret_cast<uint64_t*>(key)[0]=__builtin_bswap64(tid);
}

// This address is used to communicate that search failed
Node* nullNode=NULL;

static inline unsigned ctz(uint16_t x) {
   // Count trailing zeros, only defined for x>0
#ifdef __GNUC__
   return __builtin_ctz(x);
#else
   // Adapted from Hacker's Delight
   unsigned n=1;
   if ((x&0xFF)==0) {n+=8; x=x>>8;}
   if ((x&0x0F)==0) {n+=4; x=x>>4;}
   if ((x&0x03)==0) {n+=2; x=x>>2;}
   return n-(x&1);
#endif
}

void printkey(uint8_t *key) {
   for(int i=0; i<8; i++) {
      printf("%d ", key[i]);
   }
   printf("\n");
   return;
}

Node** findChild(Node* n,uint8_t keyByte) {
   // Find the next child for the keyByte
   switch (n->type) {
      case NodeType4: {
         // printf("node4\n");
         Node4* node=static_cast<Node4*>(n);
         for (unsigned i=0;i<node->count;i++) {
            //printf("i: %d\n", node->key[i]);
            if (node->key[i]==keyByte)
               return &node->child[i];
         }
         return &nullNode;
      }
      case NodeType16: {
         Node16* node=static_cast<Node16*>(n);
         __m128i cmp=_mm_cmpeq_epi8(_mm_set1_epi8(flipSign(keyByte)),_mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
         unsigned bitfield=_mm_movemask_epi8(cmp)&((1<<node->count)-1);
         if (bitfield)
            return &node->child[ctz(bitfield)]; else
            return &nullNode;
      }
      case NodeType48: {
         Node48* node=static_cast<Node48*>(n);
         if (node->childIndex[keyByte]!=emptyMarker)
            return &node->child[node->childIndex[keyByte]]; else
            return &nullNode;
      }
      case NodeType256: {
         Node256* node=static_cast<Node256*>(n);
         return &(node->child[keyByte]);
      }
      case NodeTypeLinear: {
         // printf("nodelinear\n");
         NodeLinear* node=static_cast<NodeLinear*>(n);
         int bucket = (int)(node->a*keyByte+node->b);
         if(bucket >= LINEAR_SIZE) {
            return &(node->child[LINEAR_SIZE-1]);
         } else if(bucket < 0) {
            return &(node->child[0]);
         } else {
            return &(node->child[bucket]);
         }
      }
   }
   throw; // Unreachable
}

Node* minimum(Node* node) {
   // Find the leaf with smallest key
   if (!node)
      return NULL;

   if (isLeaf(node))
      return node;

   switch (node->type) {
      case NodeType4: {
         Node4* n=static_cast<Node4*>(node);
         return minimum(n->child[0]);
      }
      case NodeType16: {
         Node16* n=static_cast<Node16*>(node);
         return minimum(n->child[0]);
      }
      case NodeType48: {
         Node48* n=static_cast<Node48*>(node);
         unsigned pos=0;
         while (n->childIndex[pos]==emptyMarker)
            pos++;
         return minimum(n->child[n->childIndex[pos]]);
      }
      case NodeType256: {
         Node256* n=static_cast<Node256*>(node);
         unsigned pos=0;
         while (!n->child[pos])
            pos++;
         return minimum(n->child[pos]);
      }
   }
   throw; // Unreachable
}

Node* maximum(Node* node) {
   // Find the leaf with largest key
   if (!node)
      return NULL;

   if (isLeaf(node))
      return node;

   switch (node->type) {
      case NodeType4: {
         Node4* n=static_cast<Node4*>(node);
         return maximum(n->child[n->count-1]);
      }
      case NodeType16: {
         Node16* n=static_cast<Node16*>(node);
         return maximum(n->child[n->count-1]);
      }
      case NodeType48: {
         Node48* n=static_cast<Node48*>(node);
         unsigned pos=255;
         while (n->childIndex[pos]==emptyMarker)
            pos--;
         return maximum(n->child[n->childIndex[pos]]);
      }
      case NodeType256: {
         Node256* n=static_cast<Node256*>(node);
         unsigned pos=255;
         while (!n->child[pos])
            pos--;
         return maximum(n->child[pos]);
      }
   }
   throw; // Unreachable
}

bool leafMatches(Node* leaf,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
   // Check if the key of the leaf is equal to the searched key
   if (depth!=keyLength) {
      uint8_t leafKey[maxKeyLength];
      loadKey(getLeafValue(leaf),leafKey);
      for (unsigned i=depth;i<keyLength;i++)
         if (leafKey[i]!=key[i])
            return false;
   }
   return true;
}

unsigned prefixMismatch(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength) {
   // Compare the key with the prefix of the node, return the number matching bytes
   unsigned pos;
   if (node->prefixLength>maxPrefixLength) {
      for (pos=0;pos<maxPrefixLength;pos++)
         if (key[depth+pos]!=node->prefix[pos])
            return pos;
      uint8_t minKey[maxKeyLength];
      loadKey(getLeafValue(minimum(node)),minKey);
      for (;pos<node->prefixLength;pos++)
         if (key[depth+pos]!=minKey[depth+pos])
            return pos;
   } else {
      for (pos=0;pos<node->prefixLength;pos++)
         if (key[depth+pos]!=node->prefix[pos])
            return pos;
   }
   return pos;
}


void travelNode4(Node4* node, int depth, int nodes[5], int mode);
void travelNode16(Node16* node, int depth, int nodes[5], int mode);
void travelNode48(Node48* node, int depth, int nodes[5], int mode);
void travelNode256(Node256* node, int depth, int nodes[5], int mode);
void travelNodeLinear(NodeLinear* node, int depth, int nodes[5], int mode);

void travel(Node* node, int depth, int nodes[5], int mode) {

   if (node==NULL) {
      return;
   } else if (isLeaf(node)) {
      // printf("leaf value: %d\n", getLeafValue(node));
      return;
   }

   // printf("traveling node type %d with depth %d\n", node->type, depth);
   if(mode == COUNT_NODES) nodes[node->type]++;

   switch (node->type) {
      case NodeType4:
         travelNode4(static_cast<Node4*>(node), depth, nodes, mode);
         break;
      case NodeType16:
         travelNode16(static_cast<Node16*>(node), depth, nodes, mode);
         break;
      case NodeType48:
         travelNode48(static_cast<Node48*>(node), depth, nodes, mode);
         break;
      case NodeType256:
         travelNode256(static_cast<Node256*>(node), depth, nodes, mode);
         break;
      case NodeTypeLinear:
         travelNodeLinear(static_cast<NodeLinear*>(node), depth, nodes, mode);
         break;
   }
   return;
}

void travelNode4(Node4* node, int depth, int nodes[5], int mode) {
   // printf("node4 has count %d\n", node->count);
   for(int i=0; i < node->count; i++) {
      if(mode == CHILDREN_NODES && (isLeaf(node->child[i]) || node->child[i] != NULL)) nodes[node->type]++;
      travel(node->child[i], depth+node->prefixLength, nodes, mode);
   }
}

void travelNode16(Node16* node, int depth, int nodes[5], int mode) {
   for(int i=0; i < 16; i++) {
      if(mode == CHILDREN_NODES && (isLeaf(node->child[i]) || node->child[i] != NULL)) nodes[node->type]++;
      travel(node->child[i], depth+node->prefixLength, nodes, mode);
   }
}

void travelNode48(Node48* node, int depth, int nodes[5], int mode) {
   for(int i=0; i < NODE48_SIZE; i++) {
      if(mode == CHILDREN_NODES && (isLeaf(node->child[i]) || node->child[i] != NULL)) nodes[node->type]++;
      travel(node->child[i], depth+node->prefixLength, nodes, mode);
   }
}

void travelNode256(Node256* node, int depth, int nodes[5], int mode) {
   for(int i=0; i < 256; i++) {
      if(mode == CHILDREN_NODES && (isLeaf(node->child[i]) || node->child[i] != NULL)) nodes[node->type]++;
      travel(node->child[i], depth+node->prefixLength, nodes, mode);
   }
}

void travelNodeLinear(NodeLinear* node, int depth, int nodes[5], int mode) {
   // printf("linear node found\n");
   for(int i=0; i<LINEAR_SIZE; i++) {
      if(mode == CHILDREN_NODES && (isLeaf(node->child[i]) || node->child[i] != NULL)) nodes[node->type]++;
      travel(node->child[i], depth+node->prefixLength, nodes, mode);
   }
}


void profile(Node* node) {
   // travel
   // desired output:
   // x nodes of each type
   // average use of each node
   // number of nodes at each lyaer
   // 
   int nodes[5] = {0, 0, 0, 0, 0};
   travel(node, 0, nodes, COUNT_NODES);
   printf("counting nodes\n");

   int children[5] = {0, 0, 0, 0, 0};
   travel(node, 0, children, CHILDREN_NODES);
   printf("counting children\n");

   for(int i=0; i<5; i++) {
      printf("node type %d has %d nodes and total %d children, for an average of %f children per node\n", i, nodes[i], children[i], children[i]*1.0/nodes[i]);
   }
}

Node* lookup(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
   // Find the node with a matching key, optimistic version

   bool skippedPrefix=false; // Did we optimistically skip some prefix without checking it?

   while (node!=NULL) {
      if (isLeaf(node)) {
         // printf("leaf\n");
         if (!skippedPrefix&&depth==keyLength) // No check required
            return node;

         if (depth!=keyLength) {
            // Check leaf
            uint8_t leafKey[maxKeyLength];
            loadKey(getLeafValue(node),leafKey);
            for (unsigned i=(skippedPrefix?0:depth);i<keyLength;i++)
               if (leafKey[i]!=key[i])
                  return NULL;
         }
         return node;
      }

      if (node->prefixLength) {
         // printf("prefixLength\n");
         if (node->prefixLength<maxPrefixLength) {
            for (unsigned pos=0;pos<node->prefixLength;pos++)
               if (key[depth+pos]!=node->prefix[pos])
                  return NULL;
         } else
            skippedPrefix=true;
         depth+=node->prefixLength;
      }

      // printf("depth: %d\n", depth);

      unsigned type = node->type;
      node=*findChild(node,key[depth]);
      if(type != 4) depth++;
   }

   return NULL;
}

Node* lookupPessimistic(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
   // Find the node with a matching key, alternative pessimistic version

   while (node!=NULL) {
      if (isLeaf(node)) {
         if (leafMatches(node,key,keyLength,depth,maxKeyLength))
            return node;
         return NULL;
      }

      if (prefixMismatch(node,key,depth,maxKeyLength)!=node->prefixLength)
         return NULL; else
         depth+=node->prefixLength;

      node=*findChild(node,key[depth]);
      depth++;
   }

   return NULL;
}

// Forward references
void insertNode4(Node4* node,Node** nodeRef,uint8_t keyByte,Node* child);
void insertNode16(Node16* node,Node** nodeRef,uint8_t keyByte,Node* child);
void insertNode48(Node48* node,Node** nodeRef,uint8_t keyByte,Node* child);
void insertNode256(Node256* node,Node** nodeRef,uint8_t keyByte,Node* child);

unsigned min(unsigned a,unsigned b) {
   // Helper function
   return (a<b)?a:b;
}

void copyPrefix(Node* src,Node* dst) {
   // Helper function that copies the prefix from the source to the destination node
   dst->prefixLength=src->prefixLength;
   memcpy(dst->prefix,src->prefix,min(src->prefixLength,maxPrefixLength));
}

void insert(Node* node,Node** nodeRef,uint8_t key[],unsigned depth,uintptr_t value,unsigned maxKeyLength) {
   // Insert the leaf value into the tree
   printf("depth: %d\n", depth);
   if (node==NULL) {
      *nodeRef=makeLeaf(value);
      return;
   }

   if (isLeaf(node)) {
      // Replace leaf with Node4 and store both leaves in it
      printf("insert: ");
      printkey(key);
      uint8_t existingKey[maxKeyLength];
      loadKey(getLeafValue(node),existingKey);
      unsigned newPrefixLength=0;
      while (existingKey[depth+newPrefixLength]==key[depth+newPrefixLength])
         newPrefixLength++;

      Node4* newNode=new Node4();
      newNode->prefixLength=newPrefixLength;
      memcpy(newNode->prefix,key+depth,min(newPrefixLength,maxPrefixLength));
      *nodeRef=newNode;

      printf("newPrefixLength: %d\n", newPrefixLength);
      insertNode4(newNode,nodeRef,existingKey[depth+newPrefixLength],node);
      insertNode4(newNode,nodeRef,key[depth+newPrefixLength],makeLeaf(value));
      return;
   }

   // Handle prefix of inner node
   if (node->prefixLength) {
      unsigned mismatchPos=prefixMismatch(node,key,depth,maxKeyLength);
      if (mismatchPos!=node->prefixLength) {
         // Prefix differs, create new node
         Node4* newNode=new Node4();
         *nodeRef=newNode;
         newNode->prefixLength=mismatchPos;
         memcpy(newNode->prefix,node->prefix,min(mismatchPos,maxPrefixLength));
         // Break up prefix
         if (node->prefixLength<maxPrefixLength) {
            insertNode4(newNode,nodeRef,node->prefix[mismatchPos],node);
            node->prefixLength-=(mismatchPos+1);
            memmove(node->prefix,node->prefix+mismatchPos+1,min(node->prefixLength,maxPrefixLength));
         } else {
            node->prefixLength-=(mismatchPos+1);
            uint8_t minKey[maxKeyLength];
            loadKey(getLeafValue(minimum(node)),minKey);
            insertNode4(newNode,nodeRef,minKey[depth+mismatchPos],node);
            memmove(node->prefix,minKey+depth+mismatchPos+1,min(node->prefixLength,maxPrefixLength));
         }
         insertNode4(newNode,nodeRef,key[depth+mismatchPos],makeLeaf(value));
         return;
      }
      depth+=node->prefixLength;
   }

   // Recurse
   Node** child=findChild(node,key[depth]);
   if (*child) {
      insert(*child,child,key,depth+1,value,maxKeyLength);
      return;
   }

   // Insert leaf into inner node
   Node* newNode=makeLeaf(value);
   switch (node->type) {
      case NodeType4: insertNode4(static_cast<Node4*>(node),nodeRef,key[depth],newNode); break;
      case NodeType16: insertNode16(static_cast<Node16*>(node),nodeRef,key[depth],newNode); break;
      case NodeType48: insertNode48(static_cast<Node48*>(node),nodeRef,key[depth],newNode); break;
      case NodeType256: insertNode256(static_cast<Node256*>(node),nodeRef,key[depth],newNode); break;
   }
}

void insertNode4(Node4* node,Node** nodeRef,uint8_t keyByte,Node* child) {
   // Insert leaf into inner node
   if (node->count<NODE4_SIZE) {
      // Insert element
      unsigned pos;
      for (pos=0;(pos<node->count)&&(node->key[pos]<keyByte);pos++);
      memmove(node->key+pos+1,node->key+pos,node->count-pos);
      memmove(node->child+pos+1,node->child+pos,(node->count-pos)*sizeof(uintptr_t));
      node->key[pos]=keyByte;
      node->child[pos]=child;
      node->count++;
   } else {
      // Grow to Node16
      Node16* newNode=new Node16();
      *nodeRef=newNode;
      newNode->count=NODE4_SIZE;
      copyPrefix(node,newNode);
      for (unsigned i=0;i<NODE4_SIZE;i++)
         newNode->key[i]=flipSign(node->key[i]);
      memcpy(newNode->child,node->child,node->count*sizeof(uintptr_t));
      delete node;
      return insertNode16(newNode,nodeRef,keyByte,child);
   }
}

void insertNode16(Node16* node,Node** nodeRef,uint8_t keyByte,Node* child) {
   // Insert leaf into inner node
   if (node->count<16) {
      // Insert element
      uint8_t keyByteFlipped=flipSign(keyByte);
      __m128i cmp=_mm_cmplt_epi8(_mm_set1_epi8(keyByteFlipped),_mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
      uint16_t bitfield=_mm_movemask_epi8(cmp)&(0xFFFF>>(16-node->count));
      unsigned pos=bitfield?ctz(bitfield):node->count;
      memmove(node->key+pos+1,node->key+pos,node->count-pos);
      memmove(node->child+pos+1,node->child+pos,(node->count-pos)*sizeof(uintptr_t));
      node->key[pos]=keyByteFlipped;
      node->child[pos]=child;
      node->count++;
   } else {
      // Grow to Node48
      
      Node48* newNode=new Node48();
      *nodeRef=newNode;
      memcpy(newNode->child,node->child,node->count*sizeof(uintptr_t));
      for (unsigned i=0;i<node->count;i++)
         newNode->childIndex[flipSign(node->key[i])]=i;
      copyPrefix(node,newNode);
      newNode->count=node->count;
      delete node;
      return insertNode48(newNode,nodeRef,keyByte,child);
      

      /*
      Node256* newNode=new Node256();
      *nodeRef=newNode;
      for (unsigned i=0;i<node->count;i++){
         newNode->child[flipSign(node->key[i])]=node->child[i];
      }
      copyPrefix(node,newNode);
      newNode->count=node->count;
      delete node;
      return insertNode256(newNode,nodeRef,keyByte,child);
      */
   }
}

void insertNode48(Node48* node,Node** nodeRef,uint8_t keyByte,Node* child) {
   // Insert leaf into inner node
   if (node->count<NODE48_SIZE) {
      // Insert element
      unsigned pos=node->count;
      if (node->child[pos])
         for (pos=0;node->child[pos]!=NULL;pos++);
      node->child[pos]=child;
      node->childIndex[keyByte]=pos;
      node->count++;
   } else {
      // Grow to Node256
      Node256* newNode=new Node256();
      for (unsigned i=0;i<256;i++)
         if (node->childIndex[i]!=emptyMarker)
            newNode->child[i]=node->child[node->childIndex[i]];
      newNode->count=node->count;
      copyPrefix(node,newNode);
      *nodeRef=newNode;
      delete node;
      return insertNode256(newNode,nodeRef,keyByte,child);
   }
}

void insertNode256(Node256* node,Node** nodeRef,uint8_t keyByte,Node* child) {
   // Insert leaf into inner node
   node->count++;
   node->child[keyByte]=child;
}

// Forward references
void eraseNode4(Node4* node,Node** nodeRef,Node** leafPlace);
void eraseNode16(Node16* node,Node** nodeRef,Node** leafPlace);
void eraseNode48(Node48* node,Node** nodeRef,uint8_t keyByte);
void eraseNode256(Node256* node,Node** nodeRef,uint8_t keyByte);

void erase(Node* node,Node** nodeRef,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
   // Delete a leaf from a tree

   if (!node)
      return;

   if (isLeaf(node)) {
      // Make sure we have the right leaf
      if (leafMatches(node,key,keyLength,depth,maxKeyLength))
         *nodeRef=NULL;
      return;
   }

   // Handle prefix
   if (node->prefixLength) {
      if (prefixMismatch(node,key,depth,maxKeyLength)!=node->prefixLength)
         return;
      depth+=node->prefixLength;
   }

   Node** child=findChild(node,key[depth]);
   if (isLeaf(*child)&&leafMatches(*child,key,keyLength,depth,maxKeyLength)) {
      // Leaf found, delete it in inner node
      switch (node->type) {
         case NodeType4: eraseNode4(static_cast<Node4*>(node),nodeRef,child); break;
         case NodeType16: eraseNode16(static_cast<Node16*>(node),nodeRef,child); break;
         case NodeType48: eraseNode48(static_cast<Node48*>(node),nodeRef,key[depth]); break;
         case NodeType256: eraseNode256(static_cast<Node256*>(node),nodeRef,key[depth]); break;
      }
   } else {
      //Recurse
      erase(*child,child,key,keyLength,depth+1,maxKeyLength);
   }
}

void eraseNode4(Node4* node,Node** nodeRef,Node** leafPlace) {
   // Delete leaf from inner node
   unsigned pos=leafPlace-node->child;
   memmove(node->key+pos,node->key+pos+1,node->count-pos-1);
   memmove(node->child+pos,node->child+pos+1,(node->count-pos-1)*sizeof(uintptr_t));
   node->count--;

   if (node->count==1) {
      // Get rid of one-way node
      Node* child=node->child[0];
      if (!isLeaf(child)) {
         // Concantenate prefixes
         unsigned l1=node->prefixLength;
         if (l1<maxPrefixLength) {
            node->prefix[l1]=node->key[0];
            l1++;
         }
         if (l1<maxPrefixLength) {
            unsigned l2=min(child->prefixLength,maxPrefixLength-l1);
            memcpy(node->prefix+l1,child->prefix,l2);
            l1+=l2;
         }
         // Store concantenated prefix
         memcpy(child->prefix,node->prefix,min(l1,maxPrefixLength));
         child->prefixLength+=node->prefixLength+1;
      }
      *nodeRef=child;
      delete node;
   }
}

void eraseNode16(Node16* node,Node** nodeRef,Node** leafPlace) {
   // Delete leaf from inner node
   unsigned pos=leafPlace-node->child;
   memmove(node->key+pos,node->key+pos+1,node->count-pos-1);
   memmove(node->child+pos,node->child+pos+1,(node->count-pos-1)*sizeof(uintptr_t));
   node->count--;

   if (node->count==NODE4_SIZE-1) {
      // Shrink to Node4
      Node4* newNode=new Node4();
      newNode->count=node->count;
      copyPrefix(node,newNode);
      for (unsigned i=0;i<NODE4_SIZE-1;i++)
         newNode->key[i]=flipSign(node->key[i]);
      memcpy(newNode->child,node->child,sizeof(uintptr_t)*NODE4_SIZE);
      *nodeRef=newNode;
      delete node;
   }
}

void eraseNode48(Node48* node,Node** nodeRef,uint8_t keyByte) {
   // Delete leaf from inner node
   node->child[node->childIndex[keyByte]]=NULL;
   node->childIndex[keyByte]=emptyMarker;
   node->count--;

   if (node->count==12) {
      // Shrink to Node16
      Node16 *newNode=new Node16();
      *nodeRef=newNode;
      copyPrefix(node,newNode);
      for (unsigned b=0;b<256;b++) {
         if (node->childIndex[b]!=emptyMarker) {
            newNode->key[newNode->count]=flipSign(b);
            newNode->child[newNode->count]=node->child[node->childIndex[b]];
            newNode->count++;
         }
      }
      delete node;
   }
}

void eraseNode256(Node256* node,Node** nodeRef,uint8_t keyByte) {
   // Delete leaf from inner node
   node->child[keyByte]=NULL;
   node->count--;

   /*
   if (node->count==12) {
      // Shrink to Node16
      Node16 *newNode=new Node16();
      *nodeRef=newNode;
      copyPrefix(node,newNode);
      for (unsigned b=0;b<256;b++) {
         if (node->child[b]) {
            newNode->key[newNode->count]=flipSign(b);
            newNode->child[newNode->count]=node->child[b];
            newNode->count++;
         }
      }
      delete node;
   }
   */

   if (node->count==NODE48_SIZE*3/4) {
      // Shrink to Node48
      Node48 *newNode=new Node48();
      *nodeRef=newNode;
      copyPrefix(node,newNode);
      for (unsigned b=0;b<256;b++) {
         if (node->child[b]) {
            newNode->childIndex[b]=newNode->count;
            newNode->child[newNode->count]=node->child[b];
            newNode->count++;
         }
      }
      delete node;
   }
   
}

static double gettime(void) {
  struct timeval now_tv;
  gettimeofday (&now_tv,NULL);
  return ((double)now_tv.tv_sec) + ((double)now_tv.tv_usec)/1000000.0;
}

void learn(NodeLinear* node, uint64_t* dataset, int n, unsigned depth) {
   int counts[256] = {0};
   for(int i=0; i<n; i++) {
      uint8_t key[8]; loadKey(dataset[i], key);
      counts[key[depth]]++;
   }

   // for(int i=0; i<128; i++) {
   //    printf("%d: %d\n", i, counts[i]);
   // }

   // veryyy simple linear regression code
   int s_x=0, s_y=0, s_xy=0, s_x2=0, s_y2=0;
   int bucket_size = n/LINEAR_SIZE;
   if(bucket_size == 0) bucket_size = 1;
   printf("bucket_size: %d\n", bucket_size);
   int y=0;
   for(int i=0; i<256; i++) {
      // printf("i: %d, counts[i]: %d\n", i, counts[i]);
      s_x += counts[i]*i;
      s_x2 += counts[i]*counts[i]*i;

      int i_count=counts[i];
      while(i_count > 0) {
         // printf("i_count: %d, y: %d, bucket_size: %d\n", i_count, y, bucket_size);
         if(i_count >= bucket_size) {
            s_y += bucket_size*y;
            s_y2 += bucket_size*y*y;
            s_xy += bucket_size*i*y;

            y++;
            i_count -= bucket_size;
            bucket_size -= bucket_size;
         } else {
            s_y += i_count*y;
            s_y2 += i_count*y*y;
            s_xy += i_count*i*y;

            bucket_size -= i_count;
            i_count -= i_count;
         }
         if (bucket_size == 0) {
            bucket_size = n/LINEAR_SIZE;
            if(bucket_size == 0) bucket_size = 1;
         }
      }
   }
   // printf("bucket size: %d, y: %d\n", bucket_size, y);
   // printf("s_x: %d, s_y: %d\n", s_x, s_y);
   // printf("s_x2: %d, s_y2: %d\n", s_x2, s_y2);

   double a = (n*s_xy - s_x*s_y)*1.0/(n*s_x2 - s_x*s_x);
   double b = (s_y*s_x2 - s_x*s_xy)*1.0/(n*s_x2 - s_x*s_x);
   printf("y = %fx + %f\n", a, b);
   // printf("%p\n", node->child);
   printf("test a %f, b %f\n", node->a, node->b);
   node->a = a;
   node->b = b;
   return;
}

int predict(NodeLinear* node, uint8_t* key, unsigned depth) {
   // printf("%f*%d + %f = %f\n", node->a, key[depth], node->b, node->a*key[depth] + node->b);
   int bucket = (int)(node->a*key[depth] + node->b);
   // printf("determined bucket is %d\n", bucket);
   if(bucket < 0) return 0;
   if(bucket >= LINEAR_SIZE) return LINEAR_SIZE-1;
   return bucket;
}

void insertBulk(Node* node, Node** nodeRef, uint64_t* dataset, int n, unsigned depth) {
   for(int i=0; i<n; i++) {
      // printf("value: %d\n", dataset[i]);
   }
   printf("n: %d\n", n);
   if(n <= 8 && n > 1) {
      // printf("inserting into nodes\n");
      for (uint64_t i=1; i<n; i++) {
         uint8_t key[8];loadKey(dataset[i], key);
         for(int j=0; j<8; j++) {
            printf("%d ", key[j]);
         }
         printf("\n");
         insert(*nodeRef, nodeRef, key, depth, dataset[i], 8);
      }
      return;
   } else if (n <= 1) {
      return;
   }

   if(node == NULL) {
      NodeLinear *newNode = new NodeLinear();
      *nodeRef = newNode;
   }

   NodeLinear *linearNode = static_cast<NodeLinear*>(node);

   unsigned newPrefixLength=0;
   unsigned prefixMatch = 0;
   uint8_t prefix = 0;
   while(prefixMatch != 1 && newPrefixLength < maxPrefixLength) {
      // printf("depth: %d, prefix length: %d\n", depth, newPrefixLength);
      for(uint64_t i=0; i<n; i++) {
         uint8_t key[8];loadKey(dataset[i], key);
         if(i==0) prefix = key[depth+newPrefixLength];
         if(key[depth+newPrefixLength] != prefix) {
            prefixMatch = 1;
            break;
         }
      }

      if(prefixMatch != 1) {
         newPrefixLength++;
      }
   }

   linearNode->prefixLength=newPrefixLength;
   uint8_t key[8];loadKey(dataset[0], key);
   memcpy(linearNode->prefix,key+depth, min(newPrefixLength,maxPrefixLength));
   depth+=linearNode->prefixLength;

   learn(linearNode, dataset, n, depth);

   int bucket_counts[LINEAR_SIZE] = {0};

   for(int i=0; i<n; i++) {
      uint8_t key[8]; loadKey(dataset[i], key);
      // printf("predicting...\n");
      int bucket = predict(linearNode, key, depth);
      // printf("bucket prediction is %d\n", bucket);
      bucket_counts[bucket]++;
   }

   for(int i=0; i<LINEAR_SIZE; i++) {
      // printf("bucket %d has count %d\n", i, bucket_counts[i]);
   }

   for(int i=0; i<LINEAR_SIZE; i++) {
      uint64_t *bucket_data = new uint64_t[bucket_counts[i]];
      int idx = 0;
      for(int j=0; j<n; j++) {
         uint8_t key[8];loadKey(dataset[j], key);
         if(i == predict(linearNode, key, depth)) {
            bucket_data[idx] = dataset[j];
            idx++;
         }
      }
      if(bucket_counts[i] > 8) {
         linearNode->child[i] = static_cast<Node*>(new NodeLinear());
         insertBulk(linearNode->child[i], &linearNode->child[i], bucket_data, bucket_counts[i], depth);
      } else if (bucket_counts[i] != 0){
         // printf("pre-inserting %d\n", bucket_data[0]);
         linearNode->child[i] = makeLeaf(bucket_data[0]);
         insertBulk(linearNode->child[i], &linearNode->child[i], bucket_data, bucket_counts[i], depth);
      }
   }
   return;
}

int main(int argc,char** argv) {
   if (argc!=3) {
      printf("usage: %s n 0|1|2\nn: number of keys\n0: sorted keys\n1: dense keys\n2: sparse keys\n", argv[0]);
      return 1;
   }

   uint64_t n=atoi(argv[1]);
   uint64_t* keys=new uint64_t[n];

   // Generate keys
   for (uint64_t i=0;i<n;i++)
      // dense, sorted
      keys[i]=i+1;
   if (atoi(argv[2])==1)
      // dense, random
      std::random_shuffle(keys,keys+n);
   if (atoi(argv[2])==2)
      // "pseudo-sparse" (the most-significant leaf bit gets lost)
      for (uint64_t i=0;i<n;i++)
         keys[i]=(static_cast<uint64_t>(rand())<<32) | static_cast<uint64_t>(rand());

   // Build tree
   double start = gettime();
   Node* tree=NULL;
   if(n > 8) tree = static_cast<Node*>(new NodeLinear());
   printf("%d\n", isLeaf(tree));
   // for (uint64_t i=0;i<n;i++) {
   //    uint8_t key[8];loadKey(keys[i],key);
   //    insert(tree,&tree,key,0,keys[i],8);
   // }
   insertBulk(tree, &tree, keys, n, 0);
   printf("is leaf: %d\n", isLeaf(tree));
   printf("insert,%ld,%f\n",n,(n/1000000.0)/(gettime()-start));
   profile(tree);

   // Repeat lookup for small trees to get reproducable results
   uint64_t repeat=10000000/n;
   if (repeat<1)
      repeat=1;
   start = gettime();
   for (uint64_t r=0;r<repeat;r++) {
      for (uint64_t i=0;i<n;i++) {
         uint8_t key[8];loadKey(keys[i],key);
         Node* leaf=lookup(tree,key,8,0,8);
         // printf("%p\n", leaf);
         assert(isLeaf(leaf));
         // printf("%d %d\n", getLeafValue(leaf), keys[i]);
         assert(getLeafValue(leaf)==keys[i]);
      }
   }
   printf("lookup,%ld,%f\n",n,(n*repeat/1000000.0)/(gettime()-start));

   start = gettime();
   for (uint64_t i=0;i<n;i++) {
      uint8_t key[8];loadKey(keys[i],key);
      erase(tree,&tree,key,8,0,8);
   }
   printf("erase,%ld,%f\n",n,(n/1000000.0)/(gettime()-start));
   assert(tree==NULL);

   return 0;
}
