async function lineDetectNN(){ //Uses far too small of a dataset, doesnt work really.s
    const neural_network = await buildTrainingDataLDNN(); 

    await test_model(neural_network); 
}

async function test_model(neural_network){
    const test_x = document.getElementById("userInput").value.split(",");
    console.log(test_x);


    const tensor_test_x = tf.tensor4d(test_x, [1,4,4,1], "int32");

    const prediction = await neural_network.predict(tensor_test_x).data(); 

    console.log(prediction); 
}

async function buildTrainingDataLDNN(){
    const neural_network = tf.sequential(); 
    const trainX = tf.tensor4d([
        0, 0, 0, 0, 
        0, 0, 1, 0, 
        1, 0, 0, 0, 
        0, 0, 0, 0,
        
        0, 0, 0, 1,
        0, 0, 0, 1,
        0, 0, 0, 1,
        0, 0, 0, 1,

        0, 0, 0, 0, 
        0, 0, 0, 0,
        0, 0, 0, 0,
        1, 1, 1, 1, 

        1, 1, 1, 1, 
        0, 1, 0, 0, 
        0, 1, 0, 0,
        0, 1, 0, 0, 

        0, 1, 0, 0, 
        0, 1, 0, 0,
        1, 1, 1, 1,
        0, 1, 0, 0,

        1, 1, 1, 1,
        0, 1, 0, 0, 
        1, 1, 1, 1,
        0, 1, 0, 0, 

        0, 1, 0, 0, 
        0, 0, 0, 1, 
        0, 0, 0, 0, 
        1, 0, 0, 1, 

        1, 1, 1, 1,
        0, 0, 1, 0, 
        0, 0, 1, 0, 
        1, 1, 1, 1,
    ], shape=[8,4,4,1], "int32");

    const trainY = tf.tensor2d([

        1, 0, 0, 0, 0, 0, 0, 0, //Image has 0 lines
        0, 1, 0, 0, 0, 0, 0, 0, //Image has 1 line
        0, 1, 0, 0, 0, 0, 0, 0, //Image has 1 line
        0, 0, 1, 0, 0, 0, 0, 0, //Image has 2 lines
        0, 0, 1, 0, 0, 0, 0, 0, //Image has 2 lines
        0, 0, 0, 1, 0, 0, 0, 0, //Image has 3 lines
        1, 0, 0, 0, 0, 0, 0, 0, //Image has 0 lines
        0, 0, 0, 1, 0, 0, 0, 0, //Image has 3 lines
    ], shape =[8,8], "int32");


    neural_network.add(tf.layers.conv2d({ //Conv2d reduces the amount of feautures in an images, only keeping the most important features 
        inputShape: [4, 4, 1], 
        kernelSize: 2,                     //Looks at a subsection of an image, and grabs the most important features 
        filters: 4,                        //How many kernals will there be in the image
        strides: 1,                        //How far show the kernal window slide 
        activation: "relu", 
        kernalInitializer: "varianceScaling"
    }));

    neural_network.add(tf.layers.conv2d({ //Gradually reduce the dimentionallity of the image again  
        kernelSize: 2,                     
        filters: 4,                        
        strides: 1,                        
        activation: "relu", 
        kernalInitializer: "varianceScaling"
    }));

    neural_network.add(tf.layers.flatten()); 

    neural_network.add(tf.layers.dense({
        units: 30, 
        kernelInitializer: "varianceScaling", 
        activation: "softmax", //Used in convolution networks
        
    })); 

    neural_network.add(tf.layers.dense({ //Reduce the units to match the number of bits in our labels 
        units: 8, 
        activation: "linear",
    }));

    neural_network.compile({
        optimizer: "adam", 
        loss: "meanSquaredError"
    });


    

    const epochs = 500; 
    for(let i = 0; i < 1; i++){

        var results = await neural_network.fit(trainX, trainY, {
            epochs: epochs
        }); 

        console.log("epochs: ", i);

        console.log(results.history);
    }

   
}