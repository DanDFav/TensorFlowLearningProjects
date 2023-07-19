async function oneHotEncoding(){
    
    const neural_network = await oneHotBuildNN();

    const test_x = [0,0,0,0,0,1,0,0];
    const tensor_test_x = tf.tensor2d(test_x, shape=[1,8], "int32");
    
    const prediction = await neural_network.predict(tensor_test_x).data();
    console.log(prediction);

}


async function oneHotBuildNN(){
    const neural_network = tf.sequential(); 
    const learning_rate = 0.1; 


    const train_x = tf.tensor2d([
        0, 0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0
    ], shape = [8, 8], "int32");

    const train_y = tf.tensor2d([
        0, 1, 0, 0, 
        0, 1, 0, 0, 
        0, 0, 1, 0, 
        0, 0, 1, 0,  
        1, 0, 0, 0, 
        0, 0, 0, 1,
        0, 0, 0, 1, 
        1, 0, 0, 0,
    ], shape=[8,4], "int32");

    
    neural_network.add(tf.layers.dense({
        units: 30, 
        activation: "relu", 
        inputShape: [8]
    }));

    neural_network.add(tf.layers.dense({
        units: 4, //For the Final layer we want the number of units to equal the number of classes
        activation: "sigmoid",
    }));

    neural_network.compile({
        optimizer: tf.train.sgd(learning_rate), 
        loss: "meanSquaredError"
    });


    await neural_network.fit(train_x, train_y, {
        batchSize: 64, 
        epochs: 1000, 
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                //console.log("Epoch"); 
                console.log("Epoch: " + epoch);
                console.log(logs);
            }
        }
    }); 

    return neural_network; 
}