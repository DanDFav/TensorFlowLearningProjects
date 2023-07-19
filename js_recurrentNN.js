async function recurrentNN(){
    recurrentNNTestData(); 
}


async function recurrentNNTestData(){
 
    const sequenceLength = 9; 

    const neural_network = tf.sequential(); 

    neural_network.add(tf.layers.lstm({
        units: 8, 
        inputShape: [sequenceLength,2],
    }));

    neural_network.add(tf.layers.dense({
        units: 1, 
        activation: "sigmoid", 
    }));


    const learningRate = 3e-4; 
    neural_network.compile({
        loss: "binaryCrossentropy", 
        optimizer: tf.train.adam(learningRate), 
        metrics: ["acc"]
    });

    console.log(neural_network); 

    const [x_train, y_train] = generateDataSet(400, sequenceLength);
    await neural_network.fit(x_train, y_train, {
        epochs: 20, 
        validationSplit: 0.2, 
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log("Epoch " + epoch); 
                console.log(logs);
                console.log("Accuracy " + logs.acc);
                console.log("Validation Accuracy " + logs.val_acc);
            }
        }
    });
}   


function generateSequence(sequenceLength){
    const sequence = [];

    let currentValue = -1; 
    let label = 0; 

    for(let i = 0; i < sequenceLength; ++i){
        const value = Math.random() > 0.5 ? 0 : 1; 
        sequence.push(value);
        
        if(currentValue == value){
            repeatingSubSequence++; 
        } else {
            currentValue = value; 
            repeatingSubSequence = 1; 
        }

        if(repeatingSubSequence >= 3){
            label = 1; 
        }

    }

    return [sequence, label];
}

function generateDataSet(numberOfSamples, sequenceLength){

    const sequencesBuffer = tf.buffer([numberOfSamples, sequenceLength, 2]); 
    const labelsBuffer = tf.buffer([numberOfSamples, 1]);

    for(let i = 0; i < numberOfSamples; ++i){

        const [sequence, label] = generateSequence(sequenceLength);

        

        for(let sequenceIndex = 0; sequenceIndex < sequenceLength; ++sequenceIndex){
            sequencesBuffer.set(1,i,sequenceIndex, sequence[sequenceIndex])
        }

        labelsBuffer.set(label, i, 0);
       
    }

    return [sequencesBuffer.toTensor(), labelsBuffer.toTensor()];

}