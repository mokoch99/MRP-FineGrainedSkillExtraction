<template>
  <div class="vue-tempalte">
    <div>
    <!-- Navigation -->
    <nav class="navbar shadow bg-white rounded justify-content-between flex-nowrap flex-row fixed-top" style="margin-top:10px;margin-left:10px;margin-right:10px;">
        <h1 class="txt" style="margin-bottom:0px;margin-left:35px;">Skill<b style="font-size:43px;color:#EB4336;">um</b></h1>
        <a href="https://www.maastrichtuniversity.nl"><img src="./um.svg" width="250px" height="100px" style="margin:-25px 21px -25px -25px;"></a>
    </nav>
    <!-- Main -->
    <div id="container">
    <div class="row" style="margin-top:7%;">
    <div class="column" style="width:9%"></div>
    <div class="column" style="width:38%">
      <div class="App">
      <div class="vertical-center">
        <div class="inner-block" style="width:100%;height:100%;">
          <div class="vue-tempalte" style="height:420px">
              <h3>Try Skillum</h3>
                <div>
                  <b-tabs active-nav-item-class="font-weight-bold text-uppercase" active-tab-class="font-weight-bold text-success" content-class="mt-3" style="margin-left:5px;">
                    <b-tab title="Input">
                      <form>
                      <div>
                        <b-form-textarea id="textarea" v-model="text" placeholder="Example: Dear Mr Smith, I am writing you regarding the posting for the Data Scientist position. The nature of my degree M.s. in Artificial Intelligence at the University of Maastricht has prepared me for this position. It involved a great deal of independent research, requiring initiative, self-motivation and creativity in a wide range of different challenging projects. Beside my studies I gained a lot of practical experience in Python and acquired knowledge in state of the art machine learning topics. My programming skills … Thank you for taking the time to consider this application. I am looking forward to an invitation for an interview. Yours sincerely Thomas A. Anderson" rows="9" max-rows="9"></b-form-textarea>
                      </div>
                      <button @click="getskills" type="button" class="btn btn-dark btn-lg btn-block" style="margin-top:10px;">Submit</button>
                    </form>
                    </b-tab>
                    <b-tab title="Model">
                      <div>
                        <b-button @click="setmodel(1)" block variant="primary" class="btn btn-dark btn-lg btn-block" style="margin-top:7%;">Bidirectional LSTM</b-button>
                        <b-button @click="setmodel(2)" block variant="primary" class="btn btn-dark btn-lg btn-block">Support Vector Machine</b-button>
                        <b-button @click="setmodel(3)" block variant="primary" class="btn btn-dark btn-lg btn-block">Transformer</b-button>
                        <b-button @click="setmodel(4)" block variant="primary" class="btn btn-dark btn-lg btn-block">Random Forrest</b-button>
                        <b-button @click="setmodel(5)" block variant="primary" class="btn btn-dark btn-lg btn-block">Logistic Regression</b-button>
                      </div>
                    </b-tab>
                    <b-tab title="Settings">
                      <div style="color:black;margin-top:10px;">
                        &nbsp;Threshold:
                        <b-form-input v-model="treshold" placeholder="ts"></b-form-input>
                        <hr>
                        &nbsp;Learning rate:
                        <b-form-input v-model="training_epochs" placeholder="α"></b-form-input>
                        <hr>
                        &nbsp;Label:
                        <b-form-input v-model="training_label" placeholder="y"></b-form-input>
                      </div>
                    </b-tab>
                    <b-tab title="Filter">
                      <b-form-textarea id="textarea" v-model="filter" placeholder="To filter skills, please seperate each by a comma.            E.g.: marketing, responsive, analytical thinking, ...         (not working at the moment)" rows="10" max-rows="10"></b-form-textarea>
                    </b-tab>
                  </b-tabs>
                </div>
                <div>
            </div>
          </div>
        </div>
      </div>
    </div>
    </div>
    <div class="column" style="width:6%"></div>
    <div class="column" style="width:38%">
      <div class="App">
      <div class="vertical-center">
        <div class="inner-block" style="width:100%;height:100%;">
          <div class="vue-tempalte">
            <h3>Detected Skills</h3>
            <div class="lds-dual-ring" v-if="loading" style="margin-left:40%;margin-top:20%;white-space-collapse: discard;"></div>
            <div v-else style="white-space-collapse: discard;">
              <p v-for="item in response" :key="item.word" style="display:inline;white-space-collapse:discard;">
                <span v-if="item.type == ''" style="white-space-collapse: discard;" @click="showAlertn(item.word, item.posindex)">{{ item.word }} </span>
                <span v-else-if="item.type == 'Softskill'" style="white-space-collapse: discard;"><mark @click="showAlerts(item.type, item.word, item.layer1, item.probability, item.posindex)" style="color:#0076CC;background-color:yellow;border-radius:10px;white-space-collapse: discard;" v-b-tooltip.hover :title="item.type + ': ' + item.word +' (Category: ' + item.layer1 + ')'"><b style="white-space-collapse: discard;">{{ item.word }}</b></mark></span>
                <span v-else-if="item.type == 'Hardskill'" style="white-space-collapse: discard;"><mark @click="showAlerth(item.type, item.word, item.layer1, item.layer2, item.layer3, item.probability, item.posindex)" style="color:#EB4336;background-color:yellow;border-radius:10px;white-space-collapse: discard;" v-b-tooltip.hover :title="item.type + ': ' + item.word +' (Category: ' + item.layer2 + ')'"><b style="white-space-collapse: discard;">{{ item.word }}</b></mark></span>
                <!--
                <span v-if="item.skill == 'Hard Skill'"><mark style="color:#EB4336;background-color:yellow;border-radius:10px;" v-b-tooltip.hover :title="item.word + ': ' + item.skill +', Category: n.a.'"><b>{{ item.word }}</b></mark></span>
                <span v-else-if="item.skill == 'Soft Skill'"><mark style="color:#0076CC;background-color:yellow;border-radius:10px;" v-b-tooltip.hover :title="item.word + ': ' + item.skill +', Category: n.a.'"><b>{{ item.word }}</b></mark></span>
                <span v-else>{{ item.word }} </span>
                -->
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
    </div>
    <div class="column" style="width:9%"></div>
    </div>
    </div>
    <!-- Footer -->
    <div class="row" style="margin-top:50px">
        <div class="column" style="width:4%"></div>
        <div class="column" style="width:36%"><p><b>[Beta-version] </b> {{this.activemodel}}</p></div>
        <div class="column" style="width:32%"></div>
        <div class="column" style="width:28%"><a class="txt" style="color:black;margin-top:-25px;margin-left:150px" href="https://www.maastrichtuniversity.nl/p70070981"></a></div>
    </div>
    <img v-if="brain_gif_animation" src="https://media.giphy.com/media/SlKBbQNNZNfcPRWYW7/giphy.gif" style="margin-top:-1000px;z-index:10000000;position:relative;">
  </div>
  </div>
</template>


<script>

export default {
  name: 'App',
  data: () => ({
    name: '',
    text: '',
    activemodel: 'Bidirectional LSTM',
    activelmodeln: '',
    filter: '',
    training_epochs: '',
    training_label: '',
    treshold: '',
    response: {},
    training_response: {},
    loading: false,
    brain_gif_animation: false
  }),
  computed: {

  },
  watch: {
    name: (val) => {
      console.log(val)
    }
  },
  methods: {
      setmodel(value) {
        if (value == 1) { this.activemodel = 'Bidirectional LSTM' }
        else if (value == 2) { this.activemodel = 'Support Vector Machine'}
        else if (value == 3) { this.activemodel = 'Transformer' }
        else if (value == 4) { this.activemodel = 'Random Forrest' }
        else if (value == 5) { this.activemodel = 'Logistic Regression' }
      },
      sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
      },
      async send_train_request(word, position_index) {
        let apiUrl = "http://127.0.0.1:5000/";"http://127.0.0.1:5000/" ;"https://getskillsapi-y4qplvpc2q-ew.a.run.app"
        try {
          let response = await this.axios.post(apiUrl, {"data":this.text, "filter":this.filter, "word":word, "training_model":this.activemodel, "training":"true", "training_epochs":this.training_epochs, "treshold":this.treshold, "position_index":position_index, "prev_response":this.response, "training_label":this.training_label});
          this.response = JSON.parse(response.data);
        } catch (e) {
          console.error(e);
        }
      },
      showAlertn(word, position_index) {
      // Use sweetalert2
      this.$swal({
      title: '"' + word + '"' + '\n No skill',
      text: "Do you want Skillum to learn this?",
      confirmButtonColor: '#DEB146', 
      cancelButtonColor: '#33B049',
      cancelButtonText: 'No, thanks',
      confirmButtonText: 'Learn it',
      showCancelButton: true,
      }).then((result) => {
        if (result.isConfirmed) {
          this.send_train_request(word, position_index)
        }
      })},
      showAlerts(type, word, layer1, probability, position_index) {
      // Use sweetalert2
      this.$swal({
      title: '"' + word + '"' + '\n' + type,
      text: "Category: " + layer1 + " (" + probability + " %)",
      confirmButtonColor: '#d33', 
      cancelButtonColor: '#33B049',
      cancelButtonText: 'Okay',
      confirmButtonText: 'Discard',
      showCancelButton: true,
      }).then((result) => {
        if (result.isConfirmed) {
          this.send_train_request(word, position_index)
        }
      })},
      showAlerth(type, word, layer1, layer2, layer3, probability, position_index) {
      // Use sweetalert2
      this.$swal({
      title: '"' + word + '"' + '\n' + type,
      text: "Categories: " + layer1 + " - " + layer2 + " - " + layer3 + " (" + probability + " %)",
      confirmButtonColor: '#d33', 
      cancelButtonColor: '#33B049',
      confirmButtonText: 'Discard',
      cancelButtonText: 'Okay',
      showCancelButton: true,
      }).then((result) => {
        if (result.isConfirmed) {
          this.send_train_request(word, position_index)
        }
      })},
    async getskills() {
      this.loading = true; 
      let apiUrl = "http://127.0.0.1:5000/";"http://127.0.0.1:5000/" ;"https://getskillsapi-y4qplvpc2q-ew.a.run.app"
      try {
        let response = await this.axios.post(apiUrl, {"data":this.text, "filter":this.filter, "word":"", "training_model":this.activemodel, "training":"false", "training_epochs":this.training_epochs, "treshold":this.treshold});
        this.response = JSON.parse(response.data);
        console.log(response)
      } catch (e) {
        console.error(e);
      }
      this.loading = false;
    }
  }
}
</script>

<style>

* { margin: 0; padding: 0; }

#div1
{
    float:              left;
    margin-right:       2%;         
    margin-bottom:      10px;
    max-width:          300px;
    width:              47%;
    background-color:   #d3edff;
    border-color:       #00b5ff;
    padding-bottom:     8px;
}

#div2
{
    float:              left;
    max-width:          300px;
    width:              47%; 
}

.regularContainer
{
    float:                      left;
    padding:                    7px;
    background-color:           #ffffff;
    border:                     1px solid #00b5ff;
    width:                      784px;
}


* {
    box-sizing: border-box;
  }
  
  body {
    min-height: 100vh;
    display: flex;
    font-weight: 400;
    background-image: url('background1.jpg');
    width: 100%;
    height: 90%;
    background-size: 230% 230%;
    background-position-x: -100px;
    background-position-y: -100px;
    margin: 0px;
    padding: 0px;
    overflow-x: hidden; 
  }
  
  body,
  html,
  .App,
  .vue-tempalte,
  .vertical-center {
    width: 100%;
    height: 100%;
        margin: 0px;
    padding: 0px;
    overflow-x: hidden; 
  }

  .navbar-light {
    background-color: #ffffff;
    box-shadow: 0px 14px 80px rgba(34, 35, 58, 0.2);
  }

  .vertical-center {
    display: flex;
    text-align: left;
    justify-content: center;
    flex-direction: column;    
  }
  
  .inner-block {
    width: 450px;
    margin: auto;
    background: #ffffff;

    padding: 40px 55px 45px 55px;
    border-radius: 15px;
    transition: all .3s;
  }
  
  .vertical-center .form-control:focus {
    box-shadow: none;
    border: 1.4px solid #0065A9 ;
  }
  
  .vertical-center h3 {
    text-align: center;
    margin: 0;
    line-height: 1;
    padding-bottom: 20px;
  }
  
  label {
    font-weight: 500;
  }

.txt:hover {
    text-decoration: underline;
}

.btn {
  color: #fff;
  background-color: #0065A9;
  border-color: #0065A9; /*set the color you want here*/
  border-radius: 50px !important;
}
.btn:hover, .btn:focus, .open>.dropdown-toggle.btn {
  color: rgb(252, 252, 252);
  background-color: #21264A;
  border-color: #21264A; /*set the color you want here*/

}

.btn:active, .btn.active {
  transform: translate(0px, 1px);
  -webkit-transform: translate(0px, 1px);
  color:#34A853 !important;
}

.form-control:hover {
  border-color: #40A6F8;
}

.box{
  display: flex;
  flex-flow: row nowrap;
  justify-content: center;
  align-content: center;
  align-items:center;
}
.item{
  flex: 1 1 auto;
}

 [data-title]:hover:after {
    opacity: 1;
    transition: all 0.1s ease 0.5s;
    visibility: visible;
}
[data-title]:after {
    content: attr(data-title);
    background-color: #00FF00;
    color: #111;
    font-size: 150%;
    position: absolute;
    padding: 1px 5px 2px 5px;
    bottom: -1.6em;
    left: 100%;
    white-space: nowrap;
    box-shadow: 1px 1px 3px #222222;
    opacity: 0;
    border: 1px solid #111111;
    z-index: 100;
    visibility: hidden;
}
[data-title] {
    position: relative;
}

.lds-dual-ring {
  display: inline-block;
  width: 80px;
  height: 80px;
}
.lds-dual-ring:after {
  content: " ";
  display: block;
  width: 64px;
  height: 64px;
  margin: 8px;
  border-radius: 50%;
  border: 6px solid rgb(156, 150, 150);
  border-color: #000 transparent #000 transparent;
  animation: lds-dual-ring 1.2s linear infinite;
}
@keyframes lds-dual-ring {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

</style>
