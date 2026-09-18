<script setup lang="ts">
withDefaults(defineProps<{ name: string; size?: number }>(), { size: 16 });
</script>

<template>
  <svg
    class="app-icon"
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.8"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <template v-if="name === 'refresh'">
      <path d="M20 7v5h-5" /><path d="M4 17v-5h5" /><path d="M6.1 8.3A7 7 0 0 1 18.7 7L20 12" /><path d="M4 12l1.3 5a7 7 0 0 0 12.6-1.3" />
    </template>
    <template v-else-if="name === 'play'">
      <path d="m8 5 11 7-11 7z" />
    </template>
    <template v-else-if="name === 'pause'">
      <path d="M9 5v14M15 5v14" />
    </template>
    <template v-else-if="name === 'search'">
      <circle cx="11" cy="11" r="6.5" /><path d="m16 16 4 4" />
    </template>
    <template v-else-if="name === 'close'">
      <path d="m6 6 12 12M18 6 6 18" />
    </template>
    <template v-else-if="name === 'chevron'">
      <path d="m9 6 6 6-6 6" />
    </template>
    <template v-else-if="name === 'warning'">
      <path d="M10.2 4.4 2.8 17.2A1.2 1.2 0 0 0 3.9 19h16.2a1.2 1.2 0 0 0 1.1-1.8L13.8 4.4a2.1 2.1 0 0 0-3.6 0Z" /><path d="M12 9v4M12 16.5v.1" />
    </template>
    <template v-else-if="name === 'help'">
      <circle cx="12" cy="12" r="9" /><path d="M9.7 9a2.5 2.5 0 1 1 3.4 2.3c-.8.4-1.1.9-1.1 1.7M12 16.5v.1" />
    </template>
    <template v-else-if="name === 'trace'">
      <circle cx="5" cy="6" r="2" /><circle cx="19" cy="6" r="2" /><circle cx="12" cy="18" r="2" /><path d="M7 6h10M6 8l5 8M18 8l-5 8" />
    </template>
    <template v-else-if="name === 'fit'">
      <path d="M8 3H3v5M16 3h5v5M21 16v5h-5M8 21H3v-5" />
    </template>
    <template v-else>
      <circle cx="12" cy="12" r="8" /><path d="M12 11v5M12 8v.1" />
    </template>
  </svg>
</template>
