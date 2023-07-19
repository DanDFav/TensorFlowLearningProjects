const neuralNetwork = tf.sequential(); 
    neuralNetwork.add(tf.layers.conv2d({
        inputShape: [28,28,1], 
        kernelSize: 5, 
        filters: 8, 
        strides: 1,
        activation: "relu", 
        kernerlInitializer: "varianceScaling"
    }));

    neuralNetwork.add(tf.layers.maxPooling2d({
        poolSize: [2,2], 
        strides: [2,2]
    }));

    neuralNetwork.add(tf.layers.conv2d({
        kernelSize: 5, 
        filters: 16, 
        strides: 1,
        activation: "relu", 
        kernerlInitializer: "varianceScaling"
    }));

    neuralNetwork.add(tf.layers.maxPooling2d({
        poolSize: [2,2], 
        strides: [2,2]
    }));

    neuralNetwork.add(tf.layers.flatten());

    neuralNetwork.add(tf.layers.dense({
        units: 10, 
        kernerlInitializer: "varianceScaling", 
        activation: "relu"
    }))

    neuralNetwork.compile({
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

}

async function trainModel(){
    for(let i = 0; i < numberOfTrainingBatches; i++){
        data.getTrainingBatch(batchSize);
        let validationData;
        let testBatch; 
        if(i % testingInterval == 0){
            testBatch = data.getTestingBatch(numberOfTestingBatches); 
            validationData = [testBatch.Xtensor.reshape([numberOfTestingBatches, 28, 28, 1]), testBatch.Ytensor];
        }

    }

    
    await neuralNetwork.fit(
        

    ); 
}


let data; 
async function loadData(){
    data = new data();

    await data.load(); 

    await trainModel(); 
}

