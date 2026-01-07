#pragma once
#include <stdbool.h>

typedef char s8;
typedef short s16;
typedef int s32;
typedef long long s64;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef float f32;
typedef double f64;

#define null NULL

#define KiB(n) (n*1024)
#define MiB(n) (n*1048576)
#define GiB(n) (n*1073741824)

#define reinterpret(type, val) (*(type *)&val)
#define for_to(i, to) for (s32 i = 0; i < to; ++i)

/* == ARENAS == */
typedef struct Arena Arena;
typedef struct ArenaMarker ArenaMarker;
typedef struct PoolAllocator PoolAllocator;

typedef struct Tensor Tensor;
typedef struct TensorDim TensorDim;

struct ArenaMarker {
  ArenaMarker* prev;
};

struct Arena {
  u8* data;
  u8* cur;
  u8* end;
  ArenaMarker* last_marker;
};

Arena arena_make(u64 size);
void arena_delete(Arena* a);

u8* arena_alloc(Arena* a, u64 size); 
void arena_push_section(Arena* a);
void arena_pop_section(Arena* a);
void arena_reset(Arena* a);
void arena_free_last(Arena* a, u64 size);

#define arena_alloc_struct(a, s) (s*)arena_alloc((a), sizeof(s))
#define arena_free_struct(a, s) arena_free_last((a), sizeof(s))

/* == Pool-Allocator == */
struct PoolAllocator {
  u32 element_size;
  u8* data;  
  
  u32 first_free; // index into array, *1* based
  u32 cap;
  u32 used;
};

PoolAllocator pa_make(u32 element_size, u32 element_count);
void pa_delete(PoolAllocator* pa);

u8* pa_get(PoolAllocator* pa, u32 i);
u8* pa_alloc(PoolAllocator* pa);
void pa_free(PoolAllocator* pa, u8* ptr);
void pa_reset(PoolAllocator* pa);

/* == Tensor == */
struct TensorDim {
  u32 len;
  u32 stride;
};

#define TENSOR_REQUIRES_GRAD (1 << 0)
#define TENSOR_IS_VIEW (1 << 1)

struct Tensor {
  f32* data;
  TensorDim* dims;
  Tensor* grad;
  u16 ndim;
  u16 flags;
};

#define SHAPE(arr) arr, sizeof(arr)/sizeof(arr[0])
Tensor tensor_make(f32* data, u32* shape, u16 ndim, bool requires_grad);
Tensor tensor_zeros(u32* shape, u16 ndim, bool requires_grad);
Tensor tensor_like(f32* data, Tensor* like);
Tensor tensor_zeros_like(Tensor* ref);
Tensor tensor_view(Tensor* t, u32* shape, u16 ndim);
void tensor_free(Tensor* t);
