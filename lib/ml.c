#include "ml.h"
#include <stdlib.h>
#include <stdio.h>

/* == ARENAS == */
Arena arena_make(u64 size) {
  Arena result;
  if (size == 0) {
    size = KiB(4);
  }
  
  result.data = malloc(size);
  result.end = result.data + size;
  result.cur = result.data;
  return result;
}

void arena_delete(Arena* a) {
  free(a->data);
}

u8* arena_alloc(Arena* a, u64 size) {
  u8* result = a->cur;
  a->cur += size;
  if (a->cur > a->end) {
    // TODO: what to do about this
    printf("Failed to allocate %llu bytes, Arena too small", size);
    exit(-1);
  }
  return result;
}

void arena_push_section(Arena* a) {
  ArenaMarker* m = arena_alloc_struct(a, ArenaMarker);
  m->prev = a->last_marker;
  a->last_marker = m;
}

void arena_pop_section(Arena* a) {
  if (a->last_marker == null)
    return;

  a->cur = (u8*)a->last_marker;
  a->last_marker = a->last_marker->prev;
}

void arena_reset(Arena* a) {
  a->cur = a->data;
}

void arena_free_last(Arena* a, u64 size) {
  a->cur -= size;
  a->cur  = (u8*)max((u64)a->cur, (u64)a->data);
}

/* == Pool Allocator == */
u8* pa_get(PoolAllocator* pa, u32 i) {
  if (i > pa->cap) return null;
  
  return pa->data + (pa->element_size * i);
}

u8* pa_alloc(PoolAllocator* pa) {
  if (pa->first_free == 0) return null;
  u8* result = pa_get(pa, pa->first_free-1);
  pa->first_free = *(u32*)result;
  return result;
}

// TODO:
void pa_free(PoolAllocator* pa, u8* ptr);

void pa_reset(PoolAllocator* pa) {
  u32 last = 0;
  u8* p = pa_get(pa, pa->cap-1);
  for (u32 i = pa->cap-1; i >= 0; --i) {
    *(u32*)p = last;
    last = i;
    p -= pa->element_size;
  }
}

PoolAllocator pa_make(u32 element_size, u32 element_count) {
  PoolAllocator result;
  result.element_size = max(element_size, sizeof(u32));
  result.cap = element_count;
  result.used = 0;
  result.data = malloc(element_size*element_count);
  pa_reset(&result);
  return result;
}

void pa_delete(PoolAllocator* pa) {
  free(pa->data);
}

/* == Tensor == */
Tensor tensor_make(f32* data, u32* shape, u16 ndim, bool requires_grad);
Tensor tensor_zeros(u32* shape, u16 ndim, bool requires_grad);
Tensor tensor_like(f32* data, Tensor* like);
Tensor tensor_zeros_like(Tensor* ref);
Tensor tensor_view(Tensor* t, u32* shape, u16 ndim);

void tensor_free(Tensor* t) {
  
}
