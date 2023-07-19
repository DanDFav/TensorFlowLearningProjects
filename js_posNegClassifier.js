class Model {
    async init(){
        this.neuralNetwork = await tf.loadLayersModel("https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json"); 
        const dataset = await fetch("https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json")

        const data = await dataset.json();

        return this; 
    }
}

async function posNegMain(){
    const neuralNetwork = await new Model().init(); 
    console.log(neuralNetwork);

    getUserInput(); 

}

function getUserInput(){
    const userInput = document.getElementById("userInputPos"); 

    userInput.addEventListener("input", () => {
        const userInput = document.getElementById("userInputPos"); 

        makeAPredictions(userInput);
    });
}


