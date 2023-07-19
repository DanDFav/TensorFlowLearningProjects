const neuralNetworkHW = tf.sequential(); 
neuralNetworkHW.add(tf.layers.conv2d({
    inputShape: [28,28,1], 
    kernelSize: 5, 
    filters: 8, 
    strides: 1,
    activation: "relu", 
    kernerlInitializer: "varianceScaling"
}));

neuralNetworkHW.add(tf.layers.maxPooling2d({
    poolSize: [2,2], 
    strides: [2,2]
}));

neuralNetworkHW.add(tf.layers.conv2d({
    kernelSize: 5, 
    filters: 16, 
    strides: 1,
    activation: "relu", 
    kernerlInitializer: "varianceScaling"
}));

neuralNetworkHW.add(tf.layers.maxPooling2d({
    poolSize: [2,2], 
    strides: [2,2]
}));

neuralNetworkHW.add(tf.layers.flatten());

neuralNetworkHW.add(tf.layers.dense({
    units: 10, 
    kernerlInitializer: "varianceScaling", 
    activation: "softmax"
}))

neuralNetworkHW.compile({
    optimizer: tf.train.adam(0.1), 
    loss: "categoricalCrossentropy", 
    metrics: ["accuracy"]
});

const batchSize = 64; 
const numberOfTrainingBatches = 150;
const testingInterval = 5; 
const numberOfTestingBatches = 1000; 

async function handWritingMain(){
    await loadData(); 

    await trainModel(); 
}

async function trainModel(){
    
    for(let i = 0; i < numberOfTrainingBatches; i++){
        const trainBatch = data.getTrainingBatch(batchSize);
        let testBatch;
        let validationData; 
        if(i % testingInterval == 0){
            testBatch = data.getTestingBatch(numberOfTestingBatches); 
            validationData = [testBatch.Xtensor.reshape([numberOfTestingBatches, 28, 28, 1]), testBatch.Ytensor];
        }

        const results = await neuralNetworkHW.fit(trainBatch.Xtensor.reshape([batchSize, 28, 28, 1]), trainBatch.Ytensor, {
            batchSize: batchSize,
            validationData: validationData, 
            epochs: 5
        }); 
    
        console.log(results.history.acc[0]);
    
        trainBatch.Xtensor.dispose(); 
        trainBatch.Ytensor.dispose();
    
        if(testBatch != null){
            testBatch.Xtensor.dispose();
            testBatch.Ytensor.dispose();
        }
    
        await tf.nextFrame();

    }
}


let data; 
async function loadData(){
    data = new MNISTdata();

    await data.load(); 
}

