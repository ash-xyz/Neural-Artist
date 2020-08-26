<template>
  <div id="app">
    <Github />
    <File />
    <img alt="Vue logo" src="./assets/logo.png" />
  </div>
</template>

<script>
import Github from "./components/Github.vue";
import File from "./components/Files";
export default {
  name: "App",
  components: {
    Github,
    File
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
