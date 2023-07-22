const allFeatures = ["fixed_acidity", "volatile_acidity", "citric_acid", ]

async function wineMain(){
    const wineData = jsonWineData;
    visualiseData(wineData); 

    const wineModel = buildWineModel(); 
    tfvis.show.modelSummary({name: "summary"}, wineModel)

    const processedData = processWineData(jsonWineData); 
    const {x, y} = processedData; 

    await trainModel(wineModel, x, y); 

    await evaluateWineModel(wineModel, x, y); 
}

async function evaluateWineModel(wineModel, x, y){
    const results = await wineModel.evaluate(x, y, {batchSize: 64}); 
    console.log(results[1]);
    results[1].print(); 
}

async function trainModel(model, x, y){
    model.compile({
        optimizer: tf.train.adam(), 
        loss: 'categoricalCrossentropy', 
        metrics: ["accuracy"]
    });

    return await model.fit(x, y, {
        batchSize: 64, 
        epochs: 20, 
        shuffle: true, 
        callbacks: tfvis.show.fitCallbacks({
            name: "Training"}, ["loss", "accuracy"], 
            {
                height: 300, 
                callbacksS: ["onEpochEnd"]
            }
        )
    })
}

function processWineData(jsonWineData){
    return tf.tidy(() => {
        tf.util.shuffle(jsonWineData); 
        
        const x = extractX(jsonWineData); 
        const y = jsonWineData.map(entry => entry.quality);
        
        const xTensor = tf.tensor2d(x, [x.length, x[0].length]);
        const yTensor = tf.oneHot(tf.tensor1d(y, 'int32'), 10);
        
        const xMin = xTensor.min(); const xMax = xTensor.max();
        const yMin = yTensor.min(); const yMax = yTensor.max();

        const xNormalised = xTensor.sub(xMin).div(xMax.sub(xMin)); 
        const yNormalised = yTensor.sub(yMin).div(yMax.sub(yMin)); 

        return {
            x: xNormalised, 
            y: yNormalised, 
            xMax, xMin, yMax, yMin
        }
    });
}

function extractX(jsonWineData){
    let x = [];

    x = jsonWineData.map(entry => [
        entry.fixed_acidity, 
        entry.volatile_acidity, 
        entry.citric_acid, 
        entry.residual_sugar, 
        entry.chlorides, 
        entry.free_sulfur_dioxide, 
        entry.total_sulfur_dioxide, 
        entry.density, 
        entry.pH, 
        entry.sulphates, 
        entry.alcohol
    ]);

    return x; 
}


function buildWineModel(){
    const wineModel = tf.sequential(); 
    wineModel.add(tf.layers.dense({
        inputShape: [11], 
        units: 50, 
        useBias: true, 
        activation: "relu"
    })); 

    wineModel.add(tf.layers.dense({
        units: 30, 
        useBias: true, 
        activation: "relu"
    })); 

    wineModel.add(tf.layers.dense({
        units: 20, 
        useBias: true, 
        activation: "relu"
    }));

    wineModel.add(tf.layers.dense({
        units: 10, 
        useBias: true, 
        activation: "softmax"
    }));

    return wineModel; 
}

function visualiseData(wineData){
    let featuesToVisualise = wineData.map(entry => ({
        x: entry.fixed_acidity, 
        y: entry.quality 
    }));

    generatePlot(featuesToVisualise, "Quality vs Fixed Acidity", "Fixed Acidity", "Quality");

}

function generatePlot(data, title, xLabel, yLabel){
    tfvis.render.scatterplot(
        {name: title}, 
        {values: data}, {
            xoutput: xLabel, 
            youtput: yLabel, 
            height: 400
        }
    );
}; 