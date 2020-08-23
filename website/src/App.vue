<template>
  <div id="app">
    <Github />
    <img alt="Vue logo" src="./assets/logo.png" />
  </div>
</template>

<script>
import Github from "./components/Github.vue";
export default {
  name: "App",
  components: {
    Github
  }
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
