let tensorHousingData; 
let trainedHousingModel; 


async function runHousingModel(){
    const {inputs, labels} = tensorHousingData; 

    trainedHousingModel = buildHousingModel(); 

    await trainHousingModel(inputs, labels); 

    tfvis.show.modelSummary({
        name: "Training Sumamary"
    }, trainedHousingModel);
}   


async function trainHousingModel(x,y){
    trainedHousingModel.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError, 
        metrics: ["mse", "acc"]
    });

    const batchSize = 1; 
    const epochs = 20; 

    return await trainedHousingModel.fit(x,y,{
        batchSize, 
        epochs, 
        shuffle: true, 
        callbacks: tfvis.show.fitCallbacks(
            { name: "Model Training Results" }, 
            ["loss", "mse", "acc"], {
                height: 300, 
                callback: ["onEpochEnd"]
            }
        )
    });


}

function buildHousingModel(){
    const housingModel = tf.sequential(); 

    housingModel.add(tf.layers.dense({
        inputShape: [1], 
        units: 1, 
        useBias: true
    }))

    housingModel.add(tf.layers.dense({
        units: 1, 
        useBias: true
    }));

    return housingModel; 


}


async function loadData(){
    tensorHousingData = convertToTensor(json_housingData);  
    visualiseData(); 

}

async function visualiseData(){ 

    const data = json_housingData.map(entry => ({
        x: entry.size, 
        y: entry.price
    }));
    
    tfvis.render.scatterplot(
        {name: "Price of House vs Size"}, 
        {values: data }, 
        {xLabel: "Size", 
         yLabel: "Price"}
    ); 
}


function convertToTensor(data){
    return tf.tidy(() =>{
        
        tf.util.shuffle(data);

        const x = data.map(d => d.size);
        const y = data.map(d => d.price); 

        const xTensor = tf.tensor2d(x, [x.length, 1]);
        const yTensor = tf.tensor2d(y, [y.length, 1]);

        const xMin = xTensor.min(); 
        const xMax = xTensor.max();
        const yMin = yTensor.min();
        const yMax = yTensor.max();

        const xNormalised = xTensor.sub(xMin).div(xMax.sub(xMin));
        const yNormalised = yTensor.sub(yMin).div(yMax.sub(yMin));

        return {
            inputs: xNormalised, 
            labels: yNormalised, 
            xMax, 
            xMin, 
            yMax, 
            yMin
        };
    });
}