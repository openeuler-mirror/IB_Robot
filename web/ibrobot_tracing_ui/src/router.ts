import { defineComponent, h } from "vue";
import { createRouter, createWebHistory } from "vue-router";

const RouteOutlet = defineComponent(() => () => h("span"));

export const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: "/", redirect: "/analysis" },
    { path: "/analysis/:analysisId?", name: "analysis", component: RouteOutlet },
  ],
});
