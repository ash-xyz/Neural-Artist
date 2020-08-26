<template>
  <div class="InputField">
    <input type="file" @change="onFileSelected" />
    <button @click="onUpload">Upload</button>
    <img v-if="refresh > 0" :src="'data:image/gif;base64,' + imageAsBase64" />
    <h1>{{refresh}}</h1>
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "File",
  data() {
    return {
      imageAsBase64: "",
      refresh: 0
    };
  },
  methods: {
    onFileSelected(event) {
      this.selectedFile = event.target.files[0];
      this.model = "udnie";
    },
    onUpload() {
      const fd = new FormData();
      fd.append("image", this.selectedFile);
      fd.append("style", this.model);
      axios.post("http://127.0.0.1:5000/style", fd).then(data => {
        this.imageAsBase64 = data["status"];
        this.refresh++;
        console.log(this.refresh);
      });
    }
  }
};
</script>
