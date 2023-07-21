const allFeatures = ["fixed_acidity", "volatile_acidity", "citric_acid", ]



async function wineMain(){
    const wineData = jsonWineData;
    visualiseData(wineData); 

    const wineModel = buildWineModel(); 
    tfvis.show.modelSummary({name: "summary"}, wineModel)

    processWineData(jsonWineData); 
}

function processWineData(jsonWineData){
    return tf.tidy(() => {
        tf.util.shuffle(jsonWineData); 
        
        const x = extractX(jsonWineData); 
        const y = jsonWineData.map(entry => entry.quality);
        
        const xTensor = tf.tensor2d(x, [x.length, x[0].length]);
        const yTensor = tf.oneHot(tf.tensor1d(y, 'int32'), 10);
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
        activation: "sigmoid"
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