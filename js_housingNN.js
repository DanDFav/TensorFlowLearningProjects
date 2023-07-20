let tensorHousingData; 

async function loadData(){
    tensorHousingData = convertToTensor(json_housingData);  

    
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