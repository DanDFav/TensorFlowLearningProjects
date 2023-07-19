async function handWritingMain(){
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

}