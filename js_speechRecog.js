let modelSR; 

async function speechRecogMain(){
    modelSR = speechCommands.create("BROWSER_FFT", "directional4w"); 

    await modelSR.ensureModelLoaded();

    console.log(modelSR); 
}