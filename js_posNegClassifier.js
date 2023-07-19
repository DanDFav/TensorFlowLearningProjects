class Model {
    async init(){
        this.neuralNetwork = await tf.loadLayersModel("https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json"); 
        const dataset = await fetch("https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json")

        const data = await dataset.json();

        this.maxLength = data["max_len"];
        this.wordIndex = data["word_index"];
        this.indexFrom = data["index_from"];

        return this; 
    }
}

async function posNegMain(){
    const neuralNetwork = await new Model().init(); 
    console.log(neuralNetwork);

    getUserInput(); 

}

function makeAPrediction(text){
    const cleanText = text.trim().
        toLowerCase().
        replace(/(\.|\,|\!)/g,"").
        split(" ");

    const textBuffer = tf.buffer([1, this.maxLength], "float32");

    for(let i = 0; i < cleanText.length; ++i){
        const word = cleanText[i]; 
        textBuffer.set(this.wordIndex[word] + this.indexFrom, 0, i)

    }

    const x = textBuffer.toTensor(); 

    
}

function getUserInput(){
    const userInput = document.getElementById("userInputPos"); 

    userInput.addEventListener("input", () => {
        const userInput = document.getElementById("userInputPos"); 

        makeAPrediction(userInput.value);
    });
}