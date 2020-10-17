<template>
  <div class="InputField">
    <input type="file" @change="onFileSelected" />
    <button @click="runModel">Convert</button>
    <!--<img v-if="refresh > 0" :src="'data:image/gif;base64,' + imageAsBase64" />-->
    <h1>{{ refresh }}</h1>
  </div>
</template>

<script>
import Onnx from "onnxjs";
export default {
  name: "File",
  data() {
    return {
      refresh: 0
    };
  },
  methods: {
    onFileSelected(event) {
      this.selectedFile = event.target.files[0];
      this.refresh++;
    },
    runModel() {
      const session = new Onnx.InferenceSession({
        backendHint: "webgl"
      });

      await session.loadModel('./udnie.onnx')
    }
  }
};
</script>
